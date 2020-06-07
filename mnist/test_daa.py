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

import lib
from config import test_thresholds
from data import load_mnist
from lib.daa import LinfBLOBAttack, LinfDGFAttack
from lib.utils import (MetricsDictionary, import_klass_annotations_as_flags,
                       li_metric, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_pgd")
flags.DEFINE_string("method", "blob", "daa method")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test paramrs
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("attack_nb_restarts", "1", "number of attack restarts")

FLAGS = flags.FLAGS

daa_attacks = {'dgf': LinfDGFAttack, 'blob': LinfBLOBAttack}


def import_flags(method):
    global daa_attacks
    assert method in daa_attacks
    import_klass_annotations_as_flags(daa_attacks[method], prefix="attack_")


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_daa_{FLAGS.method}_test",
                     [__file__, lib.daa.__file__])

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
    X_shape = [FLAGS.batch_size, 28, 28, 1]
    y_shape = [FLAGS.batch_size, num_classes]
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # attack arguments
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS)
        if kwarg.startswith("attack_") and kwarg not in ['attack_nb_restarts']
    }
    daa = daa_attacks[FLAGS.method](
        lambda x: test_classifier(tf.reshape(x, [-1] + X_shape[1:]))['logits'],
        **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)

        # run attack on correctly classified points
        batch_indices = tf.range(image.shape[0])
        image_adv = tf.identity(image)
        for _ in tf.range(FLAGS.attack_nb_restarts):
            is_adv = test_classifier(image_adv)['pred'] != label
            image_adv = tf.tensor_scatter_nd_update(
                image_adv, tf.expand_dims(batch_indices[~is_adv], axis=1),
                daa.perturb(image[~is_adv], label[~is_adv]))
        assert_op = tf.Assert(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0), [image_adv])
        with tf.control_dependencies([assert_op]):
            outs_adv = test_classifier(image_adv)

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
        # NOTE: cleverhans lp-norm projection may result in numerical error
        # add small constant eps = 1e-6
        li = li_metric(image - image_adv)
        is_adv = outs_adv["pred"] != label
        for threshold in test_thresholds["li"]:
            is_adv_at_th = tf.logical_and(li <= threshold + 5e-6, is_adv)
            test_metrics["acc_li_%.2f" % threshold](~is_adv_at_th)
        test_metrics["li"](li)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics["li_corr"](li[tf.logical_and(is_corr, is_adv)])
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
                                     "epoch_li-%d.png" % batch_index)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.method)
    absl.app.run(main)
