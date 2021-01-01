from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from art.attacks.evasion import PixelAttack
from art.classifiers import TensorFlowV2Classifier
from lib.tf_utils import (MetricsDictionary, l0_metric, l0_pixel_metric,
                          l1_metric, limit_gpu_growth)
from lib.utils import (format_float, log_metrics, register_experiment_flags,
                       reset_metrics, setup_experiment)

from data import fbresnet_augmentor, get_imagenet_dataflow
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/imagenet/test_one_pixel")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
flags.DEFINE_string("data_dir", "$IMAGENET_DIR", "path to imagenet dataset")
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
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(f"madry_one_pixel_test", [__file__])

    # data
    augmentors = fbresnet_augmentor(224, training=False)
    val_ds = get_imagenet_dataflow(FLAGS.data_dir,
                                   FLAGS.batch_size,
                                   augmentors,
                                   mode='val')
    val_ds.reset_state()

    # models
    num_classes = len(TsiprasCNN.LABEL_RANGES)
    classifier = TsiprasCNN()

    @tf.function
    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 224, 224, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_tsipras(FLAGS.load_from, classifier.variables)

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

        # attack only correctly classified images
        batch_indices = tf.range(image.shape[0])
        is_corr_indx = tf.expand_dims(batch_indices[is_corr], 1)
        image_subset = tf.gather_nd(image, is_corr_indx)
        label_subset = tf.gather_nd(label, is_corr_indx)
        image_subset_int = np.cast[np.int32](image_subset * 255)
        image_adv_subset_int = a0.generate(x=image_subset_int,
                                           y=label_subset,
                                           maxiter=FLAGS.attack_iters)
        r_adv_subset = (image_adv_subset_int - image_subset_int) / float(255.0)
        image_adv = tf.tensor_scatter_nd_add(image, is_corr_indx, r_adv_subset)

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
        r = image - image_adv
        l0 = l0_metric(r)
        l0p = l0_pixel_metric(r)
        l1 = l1_metric(r)
        test_metrics["l0"](l0)
        test_metrics["l0p"](l0p)
        test_metrics["l1"](l1)
        # exclude incorrectly classified
        test_metrics["l0_corr"](l0[tf.logical_and(is_corr, is_adv)])
        test_metrics["l0p_corr"](l0p[tf.logical_and(is_corr, is_adv)])
        test_metrics["l1_corr"](l1[tf.logical_and(is_corr, is_adv)])

        is_adv_at_th = tf.logical_and(l0 <= FLAGS.attack_threshold, is_adv)
        test_metrics["acc_l0_%s" %
                     format_float(FLAGS.attack_threshold)](~is_adv_at_th)
        is_adv_at_th = tf.logical_and(l0p <= FLAGS.attack_threshold, is_adv)
        test_metrics["acc_l0p_%s" %
                     format_float(FLAGS.attack_threshold)](~is_adv_at_th)
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(val_ds, 1):
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
    limit_gpu_growth()
    absl.app.run(main)
