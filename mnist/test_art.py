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
from art.attacks import (CarliniL2Method, CarliniLInfMethod, DeepFool,
                         ElasticNet)
from art.classifiers import TensorFlowV2Classifier

from config import test_thresholds
from data import load_mnist
from lib.utils import (MetricsDictionary, import_klass_annotations_as_flags,
                       l0_metric, l1_metric, l2_metric, li_metric, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, setup_experiment)
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_art")
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
        'df': DeepFool,
        'cw': CarliniL2Method
    },
    "li": {'cw': CarliniLInfMethod},
    "l1": {
        'ead': ElasticNet
    }
}


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_{FLAGS.attack}_{FLAGS.norm}_art_test", [__file__])
    logging.getLogger().setLevel(logging.INFO)

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

    # art model wrapper
    def art_classifier(x):
        return test_classifier(x)['logits']

    art_model = TensorFlowV2Classifier(
        model=art_classifier,
        input_shape=X_shape[1:],
        nb_classes=num_classes,
        channel_index=3,
        clip_values=(0, 1))

    # attacks
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }
    attack = lp_attacks[FLAGS.norm][FLAGS.attack](art_model, **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)
        is_corr = outs["pred"] == label

        batch_indices = tf.range(image.shape[0])
        is_corr = outs['pred'] == label
        image_adv = tf.identity(image)
        image_adv = tf.tensor_scatter_nd_update(
            image_adv, tf.expand_dims(batch_indices[is_corr], axis=1),
            attack.generate(image[is_corr], label[is_corr]))
        assert tf.reduce_all(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0)), "Outside range"

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

        # measure norm
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
    assert args.norm in lp_attacks
    assert args.attack in lp_attacks[args.norm]
    import_klass_annotations_as_flags(lp_attacks[args.norm][args.attack],
                                      "attack_",
                                      include_kwargs_with_defaults=True)
    absl.app.run(main)
