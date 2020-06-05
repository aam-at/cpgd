from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from cleverhans.attacks import SparseL1Descent
from cleverhans.model import Model

from config import test_thresholds
from data import load_mnist
from lib.utils import (MetricsDictionary, import_func_annotations_as_flags,
                       l1_metric, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_pgd_l1")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
import_func_annotations_as_flags(SparseL1Descent.parse_params, prefix="attack_",
                                 exclude_args=['clip_min', 'clip_max'],
                                 include_kwargs_with_defaults=True)

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_pgd_l1_test", [__file__])

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

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 28, 28, 1])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # l1 sparse PGD
    class MadryModel(Model):
        def get_logits(self, x, **kwargs):
            return test_classifier(x, **kwargs)["logits"]

        def get_probs(self, x, **kwargs):
            return test_classifier(x, **kwargs)["prob"]

    pgd_l1 = SparseL1Descent(MadryModel())
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }
    pgd_l1.parse_params(**attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        outs = test_classifier(image)

        # run attack on correctly classified points
        batch_indices = tf.range(image.shape[0])
        is_corr = outs['pred'] == label
        image_adv = tf.identity(image)
        image_adv = tf.tensor_scatter_nd_update(
            image_adv, tf.expand_dims(batch_indices[is_corr], axis=1),
            pgd_l1.generate(image[is_corr]))
        assert_op = tf.Assert(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0), [image_adv])
        with tf.control_dependencies([assert_op]):
            outs_l1 = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l1 = acc_fn(label, outs_l1["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l1"](acc_l1)
        test_metrics["conf_l1"](outs_l1["conf"])

        # measure norm
        l1 = l1_metric(image - image_adv)
        is_adv = outs_l1["pred"] != label
        for threshold in test_thresholds["l1"]:
            is_adv_at_th = tf.logical_and(l1 <= threshold, is_adv)
            test_metrics["acc_l1_%.2f" % threshold](~is_adv_at_th)
        test_metrics["l1"](l1)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics["l1_corr"](l1[tf.logical_and(is_corr, is_adv)])
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
            save_path = os.path.join(FLAGS.samples_dir,
                                     "epoch_l1-%d.png" % batch_index)
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
            np.savez(Path(FLAGS.working_dir) / "X_adv",
                     X_adv=X_lp_all,
                     y=y_all)


if __name__ == "__main__":
    absl.app.run(main)
