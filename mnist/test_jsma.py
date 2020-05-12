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

from cleverhans.attacks.saliency_map_method import SaliencyMapMethod
from cleverhans.model import Model
from data import load_mnist
from lib.utils import (MetricsDictionary, import_kwargs_as_flags, l0_metric,
                       log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_jsma")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_float("attack_theta", 1.0, "theta for jsma")
flags.DEFINE_float("attack_gamma", 1.0, "gamma for jsma")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_jsma_test", [__file__])

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

    class MadryModel(Model):
        def get_logits(self, x, **kwargs):
            return test_classifier(x, **kwargs)["logits"]

        def get_probs(self, x, **kwargs):
            return test_classifier(x, **kwargs)["prob"]

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 28, 28, 1])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # saliency map method attack
    l0_thresholds = np.linspace(2, 100, 99)
    jsma = SaliencyMapMethod(MadryModel())
    jsma.parse_params(theta=FLAGS.attack_theta,
                      gamma=FLAGS.attack_gamma,
                      clip_min=0.0,
                      clip_max=1.0)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        indices = tf.argsort(label_onehot)[:, :-1]
        bestlp = np.inf * tf.ones(image.shape[0])
        image_adv = tf.identity(image)
        for i in tf.range(num_classes - 1):
            target_onehot = tf.one_hot(indices[:, i], num_classes)
            image_adv_i = jsma.generate(image, y_target=target_onehot)
            l0 = l0_metric(image_adv_i - image)
            image_adv = tf.where(tf.reshape(l0 < bestlp, (-1, 1, 1, 1)),
                                 image_adv_i, image_adv)
            bestlp = tf.minimum(bestlp, l0)

        outs = test_classifier(image)
        outs_l0 = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l0 = acc_fn(label, outs_l0["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_l0"](acc_l0)
        test_metrics[f"conf_l0"](outs_l0["conf"])

        # measure norm
        l0 = l0_metric(image - image_adv)
        is_adv = outs_l0["pred"] != label
        for threshold in l0_thresholds:
            is_adv_at_th = tf.logical_and(l0 <= threshold, is_adv)
            test_metrics[f"acc_l0_%.2f" % threshold](~is_adv_at_th)
        test_metrics[f"l0"](l0)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics[f"l0_corr"](l0[is_corr])

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
                                     f"epoch_l0-%d.png" % batch_index)
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
