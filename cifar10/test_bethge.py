from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from foolbox.attacks import (DatasetAttack, L0BrendelBethgeAttack,
                             L1BrendelBethgeAttack, L2BrendelBethgeAttack,
                             LinearSearchBlendedUniformNoiseAttack,
                             LinfinityBrendelBethgeAttack)
from foolbox.models import TensorFlowModel

from config import test_thresholds
from data import load_cifar10
from lib.tf_utils import (MetricsDictionary, l0_metric, l0_pixel_metric,
                          l1_metric, l2_metric, li_metric, make_input_pipeline)
from lib.utils import (import_klass_annotations_as_flags, log_metrics,
                       register_experiment_flags, reset_metrics,
                       setup_experiment)
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_brendel_lp")
flags.DEFINE_string("norm", "l1", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

FLAGS = flags.FLAGS

lp_attacks = {
    "l0": L0BrendelBethgeAttack,
    "l1": L1BrendelBethgeAttack,
    "l2": L2BrendelBethgeAttack,
    "li": LinfinityBrendelBethgeAttack,
}


def import_flags(norm):
    global lp_attacks
    assert norm in lp_attacks
    import_klass_annotations_as_flags(lp_attacks[norm], "attack_")


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_bethge_{FLAGS.norm}_test", [__file__])

    # data
    _, _, test_ds = load_cifar10(FLAGS.validation_size,
                                 data_format="NHWC",
                                 seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    classifier = MadryCNNTf(model_type=model_type)

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    fclassifier = TensorFlowModel(lambda x: test_classifier(x)["logits"],
                                  bounds=np.array((0.0, 1.0),
                                                  dtype=np.float32))

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 32, 32, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    lp_metrics = {
        "l0": l0_pixel_metric,
        "l1": l1_metric,
        "l2": l2_metric,
        "li": li_metric,
    }
    # attack arguments
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }
    b_and_b = lp_attacks[FLAGS.norm](**attack_kwargs)
    # init attacks
    a0 = LinearSearchBlendedUniformNoiseAttack()
    a0_2 = DatasetAttack()
    for image, _ in test_ds:
        a0_2.feed(fclassifier, image)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)
        is_corr = outs['pred'] == label

        # run attack on correctly classified points
        batch_indices = tf.range(image.shape[0])
        image_s = image[is_corr]
        label_s = label[is_corr]

        # get attack starting points
        x0 = a0.run(fclassifier, image_s, label_s)
        is_adv = tf.argmax(fclassifier(x0), axis=-1) != label_s
        # run dataset attack if LinearSearchBlendedUniformNoiseAttack fails
        if not tf.reduce_all(is_adv):
            x0 = tf.where(
                tf.reshape(
                    tf.argmax(fclassifier(x0), axis=-1) != label,
                    (-1, 1, 1, 1)), x0, a0_2.run(fclassifier, image_s,
                                                 label_s))
        image_adv = tf.identity(image)
        image_adv = tf.tensor_scatter_nd_update(
            image_adv, tf.expand_dims(batch_indices[is_corr], axis=1),
            b_and_b(fclassifier,
                    image_s,
                    label_s,
                    starting_points=x0,
                    epsilons=None)[0])
        # sanity check
        assert_op = tf.Assert(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0), [image_adv])
        with tf.control_dependencies([assert_op]):
            outs_adv = test_classifier(image_adv)
            is_adv = test_classifier(image_adv)['pred'] != label

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_adv = acc_fn(label, outs_adv["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_{FLAGS.norm}"](acc_adv)
        test_metrics[f"conf_{FLAGS.norm}"](outs_adv["conf"])

        # measure norm
        r = image - image_adv
        lp = lp_metrics[FLAGS.norm](r)
        l0 = l0_metric(r)
        l0p = l0_pixel_metric(r)
        l1 = l1_metric(r)
        l2 = l2_metric(r)
        li = li_metric(r)
        test_metrics["l0"](l0)
        test_metrics["l0p"](l0p)
        test_metrics["l1"](l1)
        test_metrics["l2"](l2)
        test_metrics["li"](li)
        # exclude incorrectly classified
        test_metrics["l0_corr"](l0[tf.logical_and(is_corr, is_adv)])
        test_metrics["l0p_corr"](l0p[tf.logical_and(is_corr, is_adv)])
        test_metrics["l1_corr"](l1[tf.logical_and(is_corr, is_adv)])
        test_metrics["l2_corr"](l2[tf.logical_and(is_corr, is_adv)])
        test_metrics["li_corr"](li[tf.logical_and(is_corr, is_adv)])

        # robust accuracy at threshold
        for threshold in test_thresholds[FLAGS.norm]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            test_metrics[f"acc_{FLAGS.norm}_%.2f" % threshold](~is_adv_at_th)
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(test_ds, 1):
            X_lp = test_step(image, label)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                break
    except KeyboardInterrupt:
        logging.info("Stopping after {}".format(batch_index))
    except Exception as e:
        raise
    finally:
        e = sys.exc_info()[1]
        if e is None or isinstance(e, KeyboardInterrupt):
            log_metrics(
                test_metrics,
                "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                     batch_index),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.norm)
    absl.app.run(main)
