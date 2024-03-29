from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from cleverhans.attacks import ProjectedGradientDescent, SparseL1Descent
from cleverhans.model import Model
from lib.attack_utils import cw_loss
from lib.pgd_l0 import SparseL0Descent
from lib.tf_utils import (MetricsDictionary, l0_metric, l0_pixel_metric,
                          l1_metric, l2_metric, li_metric, make_input_pipeline)
from lib.utils import (format_float, import_func_annotations_as_flags,
                       log_metrics, register_experiment_flags, reset_metrics,
                       setup_experiment)

from config import test_thresholds
from data import load_cifar10
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_pgd")
flags.DEFINE_string("norm", "lp", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test params
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("attack_nb_restarts", "1", "number of attack restarts")
flags.DEFINE_string("attack_loss", "ce", "loss for the attack")

FLAGS = flags.FLAGS

lp_attacks = {
    "l0": SparseL0Descent,
    "l1": SparseL1Descent,
    "l2": ProjectedGradientDescent,
    "li": ProjectedGradientDescent,
}


def import_flags(norm):
    global lp_attacks
    assert norm in lp_attacks
    exclude_args = ['clip_min', 'clip_max', 'rand_init']
    if norm not in ['l0', 'l1']:
        exclude_args.append('ord')
    import_func_annotations_as_flags(lp_attacks[norm].parse_params,
                                     prefix="attack_",
                                     exclude_args=exclude_args)


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_pgd_{FLAGS.norm}_test", [__file__])

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
    classifier = MadryCNNTf(model_type=model_type)

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 32, 32, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    # lp sparse PGD
    class MadryModel(Model):
        def get_logits(self, x, **kwargs):
            return test_classifier(x, **kwargs)["logits"]

        def get_probs(self, x, **kwargs):
            return test_classifier(x, **kwargs)["prob"]

    pgd = lp_attacks[FLAGS.norm](MadryModel())

    lp_metrics = {
        "l1": l1_metric,
        "l2": l2_metric,
        "li": li_metric,
        "l0": l0_metric
    }

    # attack arguments
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
        and kwarg not in ['attack_loss', 'attack_nb_restarts']
    }
    if FLAGS.norm not in ['l0', 'l1']:
        attack_kwargs['ord'] = 2 if FLAGS.norm == 'l2' else np.inf
    # select loss
    if FLAGS.attack_loss == "cw":
        attack_kwargs["loss_fn"] = lambda labels, logits: -cw_loss(
            labels, logits)
    elif FLAGS.attack_loss == "ce":
        pass
    else:
        raise ValueError(f"Invalid loss {FLAGS.attack_loss}")
    pgd.parse_params(**attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        outs = test_classifier(image)
        is_corr = test_classifier(image)['pred'] == label

        # run attack on correctly classified points
        batch_indices = tf.range(image.shape[0])
        image_adv = tf.identity(image)
        for _ in tf.range(FLAGS.attack_nb_restarts):
            is_adv = test_classifier(image_adv)['pred'] != label
            image_adv = tf.tensor_scatter_nd_update(
                image_adv, tf.expand_dims(batch_indices[~is_adv], axis=1),
                pgd.generate(image[~is_adv],
                             y=label_onehot[~is_adv],
                             clip_min=0.0,
                             clip_max=1.0,
                             rand_init=True,
                             **attack_kwargs))
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
        r = image - image_adv
        lp = lp_metrics[FLAGS.norm](r)
        l0 = l0_metric(r)
        l0p = l0_pixel_metric(r)
        l1 = l1_metric(r)
        l2 = l2_metric(r)
        li = li_metric(r)
        test_metrics["l0"](l0)
        test_metrics["l0p"](l0p)
        test_metrics["l1"](l1)
        test_metrics["l2"](l2)
        test_metrics["li"](li)
        # exclude incorrectly classified
        test_metrics["l0_corr"](l0[tf.logical_and(is_corr, is_adv)])
        test_metrics["l0p_corr"](l0p[tf.logical_and(is_corr, is_adv)])
        test_metrics["l1_corr"](l1[tf.logical_and(is_corr, is_adv)])
        test_metrics["l2_corr"](l2[tf.logical_and(is_corr, is_adv)])
        test_metrics["li_corr"](li[tf.logical_and(is_corr, is_adv)])

        # robust accuracy at threshold
        # NOTE: cleverhans lp-norm projection may result in numerical error
        # add small constant eps = 1e-6
        for threshold in test_thresholds[f"{FLAGS.norm}"]:
            is_adv_at_th = tf.logical_and(lp <= threshold + 5e-6, is_adv)
            test_metrics[f"acc_{FLAGS.norm}_%s" %
                         format_float(threshold, 4)](~is_adv_at_th)
            if FLAGS.norm == "l0":
                is_adv_at_th = tf.logical_and(l0p <= threshold, is_adv)
                test_metrics["acc_l0p_%s" %
                             format_float(threshold, 4)](~is_adv_at_th)
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
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.norm)
    absl.app.run(main)
