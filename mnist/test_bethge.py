from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
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
from data import load_mnist
from lib.utils import (MetricsDictionary, import_func_annotations_as_flags,
                       l0_metric, l1_metric, l2_metric, li_metric, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_brendel_lp")
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


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_bethge_{FLAGS.norm}_test", [__file__])

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    classifier = MadryCNN()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    fclassifier = TensorFlowModel(lambda x: test_classifier(x)["logits"],
                                  bounds=np.array((0.0, 1.0),
                                                  dtype=np.float32))

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 28, 28, 1])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    lp_metrics = {
        "l0": l0_metric,
        "l1": l1_metric,
        "l2": l2_metric,
        "li": li_metric
    }
    # attacks
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

        # run attack on correctly classified points
        batch_indices = tf.range(image.shape[0])
        is_corr = outs['pred'] == label
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
        # safety check
        assert tf.reduce_all(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0)), "Outside range"

        outs_adv = test_classifier(image_adv)

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
        lp = lp_metrics[FLAGS.norm](image - image_adv)
        is_adv = outs_adv["pred"] != label
        for threshold in test_thresholds[FLAGS.norm]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            test_metrics[f"acc_{FLAGS.norm}_%.2f" % threshold](~is_adv_at_th)
        test_metrics[f"{FLAGS.norm}"](lp)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics[f"{FLAGS.norm}_corr"](lp[tf.logical_and(is_corr, is_adv)])
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
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
            save_path = os.path.join(FLAGS.samples_dir,
                                     "epoch_orig-%d.png" % batch_index)
            save_images(image, save_path, data_format="NHWC")
            save_path = os.path.join(
                FLAGS.samples_dir, f"epoch_{FLAGS.norm}-%d.png" % batch_index)
            save_images(X_lp, save_path, data_format="NHWC")
            # save adversarial data
            X_lp_list.append(X_lp)
            y_list.append(label)
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
            X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
            y_all = tf.concat(y_list, axis=0).numpy()
            np.savez(Path(FLAGS.working_dir) / "X_adv", X_adv=X_lp_all, y=y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    assert args.norm in lp_attacks
    import_func_annotations_as_flags(lp_attacks[args.norm].__init__, "attack_")
    absl.app.run(main)