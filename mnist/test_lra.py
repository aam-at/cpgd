from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
from functools import partial

import scipy
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from lib.lra import run, find_starting_point
from lib.lra.staxmod import *
import jax

from config import test_thresholds
from data import load_mnist
from lib.utils import (MetricsDictionary, import_klass_annotations_as_flags,
                       l1_metric, l2_metric, li_metric, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, setup_experiment, AttributeDict,
                       add_default_end_points)
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_lra")
flags.DEFINE_string("attack", None, "attack class")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")

# attack parameters
flags.DEFINE_integer("attack_regions", 400, "attack regions")
flags.DEFINE_integer("attack_iterations", 500, "attack iterations")
flags.DEFINE_integer("attack_gamma", 6, "attack region selection")
flags.DEFINE_float("attack_misc_factor", 75, "misc factor")
flags.DEFINE_integer('attack_nth_likely_class_starting_point', None, "")
flags.DEFINE_bool('attack_no_linesearch', False, "")
flags.DEFINE_integer('attack_max_other_classes', None, "")
flags.DEFINE_bool('attack_no_normalization', False, "")


FLAGS = flags.FLAGS


def MadryCNN():
    return serial(
        Conv(32, (5, 5), padding='SAME'), Relu,
        MaxPool((2, 2), strides=(2, 2), padding='VALID'),
        Conv(64, (5, 5), padding='SAME'), Relu,
        MaxPool((2, 2), strides=(2, 2), padding='VALID'),
        Flatten,
        Dense(1024), Relu,
        Dense(10))


def load_params(load_from):
    mapping = [
        ("A0", "A1"),
        (),
        (),
        ("A2", "A3"),
        (),
        (),
        (),
        ("A4", "A5"),
        (),
        ("A6", "A7"),
    ]
    w = scipy.io.loadmat(load_from)
    params = []
    for map_sublist in mapping:
        if len(map_sublist) == 0:
            params.append(())
        else:
            params.append((w[map_sublist[0]], w[map_sublist[1]]))
    return jax.tree_map(jax.device_put, params)


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_lra_test", [__file__])

    # data
    train_ds, _, test_ds = load_mnist(0,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    init_model, apply_model = MadryCNN()
    output_shape, _ = init_model((-1, 28, 28, 1))

    # load classifier params
    params = load_params(FLAGS.load_from)
    predict = jax.jit(partial(apply_model, params))

    # attacks
    attack_args = AttributeDict({
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    })
    attack_args.accuracy = False
    attack_args.image = 0

    find_starting_point_2 = partial(find_starting_point, train_ds[0], train_ds[1])

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        logits = predict(image.numpy())[0]
        outs = add_default_end_points({'logits': (jax.device_get(logits))})

        # attack only correctly classified examples
        batch_indices = tf.range(image.shape[0])
        is_corr = outs['pred'] == label
        image_adv = tf.identity(image)
        for indx in batch_indices[is_corr]:
            image_i = tf.expand_dims(image[indx], 0).numpy()
            label_i = tf.expand_dims(label[indx], 0).numpy()
            image_adv_jax = run(num_classes, apply_model, params,
                                image_i, label_i,
                                find_starting_point_2, attack_args)
            image_adv = tf.tensor_scatter_nd_update(
                image_adv, tf.expand_dims([indx], axis=1),
                jax.device_get(image_adv_jax))

        # safety check
        assert tf.reduce_all(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0)), "Outside range"

        logits_adv = predict(image_adv.numpy())[0]
        outs_adv = add_default_end_points({'logits': (jax.device_get(logits_adv))})

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_adv = acc_fn(label, outs_adv["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l2"](acc_adv)
        test_metrics["conf_l2"](outs_adv["conf"])

        # measure norm
        l2 = l2_metric(image - image_adv)
        is_adv = outs_adv["pred"] != label
        for threshold in test_thresholds["l2"]:
            is_adv_at_th = tf.logical_and(l2 <= threshold, is_adv)
            test_metrics["acc_l2_%.2f" % threshold](~is_adv_at_th)
        test_metrics["l2"](l2)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics["l2_corr"](l2[tf.logical_and(is_corr, is_adv)])
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
                FLAGS.samples_dir, f"epoch_l2-%d.png" % batch_index)
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
