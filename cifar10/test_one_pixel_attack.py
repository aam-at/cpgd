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
from art.attacks import PixelAttack
from art.classifiers import TensorFlowV2Classifier

from data import load_cifar10
from lib.utils import (MetricsDictionary, l0_metric, l0_pixel_metric, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_one_pixel")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("attack_threshold", 1, "pixel attack threshold")
flags.DEFINE_integer("attack_iters", 100, "number of attack iterations")
flags.DEFINE_integer("attack_es", 1, "cmaes or dae")
flags.DEFINE_bool("attack_verbose", False, "verbose?")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_one_pixel_test", [__file__])

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
    classifier = MadryCNN(model_type=model_type)

    @tf.function
    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 32, 32, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    # one pixel attack
    def art_classifier(x):
        assert x.max() > 1 and x.max() <= 255
        x = tf.cast(x / 255.0, tf.float32)
        return test_classifier(x)['logits']

    art_model = TensorFlowV2Classifier(
        model=art_classifier,
        input_shape=X_shape[1:],
        nb_classes=num_classes,
        channel_index=3,
        clip_values=(0, 1))
    a0 = PixelAttack(art_model,
                     th=FLAGS.attack_threshold,
                     es=FLAGS.attack_es,
                     verbose=FLAGS.attack_verbose)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)
        is_corr = outs['pred'] == label

        image_int = np.cast[np.int32](image * 255)
        image_adv = np.cast[np.float32](
            a0.generate(x=image_int, y=label, maxiter=FLAGS.attack_iters) /
            255.0)
        outs_l0 = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l0 = acc_fn(label, outs_l0["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l0"](acc_l0)
        test_metrics["conf_l0"](outs_l0["conf"])

        # measure norm
        l0 = l0_pixel_metric(image - image_adv, art_model.channel_index)
        is_adv = outs_l0["pred"] != label
        is_adv_at_th = tf.logical_and(l0 <= FLAGS.attack_threshold, is_adv)
        test_metrics["acc_l0_%.2f" % FLAGS.attack_threshold](~is_adv_at_th)
        test_metrics["l0"](l0)
        test_metrics["l0_all"](l0_metric(image - image_adv))
        # exclude incorrectly classified
        test_metrics["l0_corr"](l0[tf.logical_and(is_corr, is_adv)])

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
