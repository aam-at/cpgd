from __future__ import absolute_import, division, print_function

import glob
import logging
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from lib.tf_utils import (MetricsDictionary, l0_metric, l1_metric, l2_metric,
                          li_metric, make_input_pipeline)
from lib.utils import (format_float, log_metrics, register_experiment_flags,
                       reset_metrics, setup_experiment)

from config import test_thresholds
from data import load_mnist
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_combined")
flags.DEFINE_string("norm", None, "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
flags.DEFINE_list("load_list", None, "List of directories to load saved attack from")
# test params
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 500, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    assert FLAGS.norm is not None
    setup_experiment(f"madry_pgd_{FLAGS.norm}_test", [__file__])

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    X = test_ds[0]

    # models
    num_classes = 10
    classifier = MadryCNNTf()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

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

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, image_adv, label):
        outs = test_classifier(image)
        is_corr = test_classifier(image)['pred'] == label

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
        test_metrics["acc_adv"](acc_adv)
        test_metrics["conf_adv"](outs_adv["conf"])

        # measure norm
        lp = lp_metrics[FLAGS.norm](image - image_adv)
        test_metrics[f"{FLAGS.norm}"](lp)
        # exclude incorrectly classified
        test_metrics[f"{FLAGS.norm}_corr"](lp[tf.logical_and(is_corr, is_adv)])

        # robust accuracy at threshold
        # NOTE: cleverhans lp-norm projection may result in numerical error
        # add small constant eps = 1e-6
        for threshold in test_thresholds[f"{FLAGS.norm}"]:
            is_adv_at_th = tf.logical_and(lp <= threshold + 5e-6, is_adv)
            test_metrics[f"acc_{FLAGS.norm}_%s" % format_float(threshold)](~is_adv_at_th)
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        # select minimum perturbation from multiple saved attacks
        X_adv = X.copy()
        rnorm = np.inf * np.ones(X.shape[0])
        for load_regexp in FLAGS.load_list:
            for load_file in Path(load_regexp).rglob("*.npy"):
                X_adv_l = np.load(load_file).reshape(X.shape)
                rnorm2 = lp_metrics[FLAGS.norm](tf.convert_to_tensor(X - X_adv_l)).numpy()
                with tf.device("/cpu"):
                    is_adv = test_classifier(X_adv_l)['pred'] != test_ds[1]
                    assert tf.reduce_all(is_adv)
                if X_adv is None:
                    X_adv = X_adv_l
                    rnorm = rnorm2
                else:
                    X_adv[rnorm2 < rnorm] = X_adv_l[rnorm2 < rnorm]
                    rnorm = np.minimum(rnorm, rnorm2)

        # combine datasets
        test_ds = tf.data.Dataset.from_tensor_slices((test_ds[0], X_adv, test_ds[1]))
        test_ds = make_input_pipeline(test_ds,
                                      shuffle=False,
                                      batch_size=FLAGS.batch_size)
        for batch_index, (image, image_adv, label) in enumerate(test_ds, 1):
            X_lp = test_step(image, image_adv, label)
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
    absl.app.run(main)
