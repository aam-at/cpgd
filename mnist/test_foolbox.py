from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys
import time

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from foolbox.attacks import (DDNAttack, EADAttack, L2CarliniWagnerAttack,
                             L2DeepFoolAttack, LinfDeepFoolAttack,
                             NewtonFoolAttack)
from foolbox.models import TensorFlowModel

from config import test_thresholds
from data import load_mnist
from lib.tf_utils import (MetricsDictionary, l0_metric, l1_metric, l2_metric,
                          li_metric, make_input_pipeline)
from lib.utils import (import_klass_annotations_as_flags, log_metrics,
                       register_experiment_flags, reset_metrics,
                       setup_experiment)
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_foolbox")
flags.DEFINE_string("attack", None, "attack class")
flags.DEFINE_string("norm", "l2", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
FLAGS = flags.FLAGS

lp_attacks = {
    "l2": {
        'df': L2DeepFoolAttack,
        'ddn': DDNAttack,
        'cw': L2CarliniWagnerAttack,
        'newton': NewtonFoolAttack
    },
    "li": {
        'df': LinfDeepFoolAttack,
    },
    "l1": {
        'ead': EADAttack
    }
}


def import_flags(norm, attack):
    global lp_attacks
    assert norm in lp_attacks
    assert attack in lp_attacks[norm]
    import_klass_annotations_as_flags(lp_attacks[norm][attack],
                                      prefix="attack_")
    if attack == 'ead':
        flags.DEFINE_string("attack_decision_rule", "L1", "")


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_foolbox_{FLAGS.attack}_{FLAGS.norm}_test", [__file__])

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
    classifier = MadryCNNTf()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    fclassifier = TensorFlowModel(lambda x: test_classifier(x)["logits"],
                                  bounds=np.array((0.0, 1.0),
                                                  dtype=np.float32))

    lp_metrics = {
        "l2": l2_metric,
        "l1": l1_metric,
        "li": li_metric,
    }
    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 28, 28, 1])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # attacks
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }
    attack = lp_attacks[FLAGS.norm][FLAGS.attack](**attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)

        batch_indices = tf.range(image.shape[0])
        is_corr = outs['pred'] == label
        image_adv = tf.identity(image)
        image_adv = tf.tensor_scatter_nd_update(
            image_adv, tf.expand_dims(batch_indices[is_corr], axis=1),
            attack.run(fclassifier, image[is_corr], label[is_corr]))
        # sanity check
        assert_op = tf.Assert(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0), [image_adv])
        with tf.control_dependencies([assert_op]):
            outs_adv = test_classifier(image_adv)
            is_adv = outs_adv["pred"] != label

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

        r = image - image_adv
        lp = lp_metrics[FLAGS.norm](r)
        l0 = l0_metric(r)
        l1 = l1_metric(r)
        l2 = l2_metric(r)
        li = li_metric(r)
        test_metrics["l0"](l0)
        test_metrics["l1"](l1)
        test_metrics["l2"](l2)
        test_metrics["li"](li)
        # exclude incorrectly classified
        test_metrics["l0_corr"](l0[tf.logical_and(is_corr, is_adv)])
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
    parser.add_argument("--attack", default=None, type=str)
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.norm, args.attack)
    absl.app.run(main)
