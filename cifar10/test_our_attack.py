from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import time
import traceback
from pathlib import Path

import absl
import lib
import numpy as np
import tensorflow as tf
from absl import flags
from lib.attack_l0 import ClassConstrainedProximalL0Attack
from lib.attack_l1 import (ClassConstrainedL1Attack,
                           ClassConstrainedProximalL1Attack)
from lib.attack_l2 import (ClassConstrainedL2Attack,
                           ClassConstrainedProximalL2Attack)
from lib.attack_li import ClassConstrainedProximalLiAttack
from lib.attack_utils import AttackOptimizationLoop
from lib.tf_utils import (MetricsDictionary, l0_metric, l0_pixel_metric,
                          l1_metric, l2_metric, li_metric, make_input_pipeline)
from lib.utils import (format_float, import_klass_annotations_as_flags,
                       log_metrics, register_experiment_flags, reset_metrics,
                       setup_experiment)
from tensorboard.plugins.hparams import api as hp

from config import test_thresholds
from data import load_cifar10
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_lp")
flags.DEFINE_string("attack", None,
                    "choice of the attack ('l0', 'l1', 'l2', 'l2g', 'li')")
flags.DEFINE_bool("attack_save", False, "True if save results of the attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
import_klass_annotations_as_flags(AttackOptimizationLoop, "attack_loop_")

FLAGS = flags.FLAGS

lp_attacks = {
    "l0": ("l0", ClassConstrainedProximalL0Attack),
    "l1": ("l1", ClassConstrainedProximalL1Attack),
    "l1g": ("l1", ClassConstrainedL1Attack),
    "l2": ("l2", ClassConstrainedProximalL2Attack),
    "l2g": ("l2", ClassConstrainedL2Attack),
    "li": ("li", ClassConstrainedProximalLiAttack),
}


def import_flags(attack):
    assert attack in lp_attacks
    attack_klass = lp_attacks[attack][1]
    import_klass_annotations_as_flags(attack_klass, "attack_")


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    norm, attack_klass = lp_attacks[FLAGS.attack]

    assert FLAGS.load_from is not None
    setup_experiment(
        f"madry_{norm}_test",
        [
            __file__,
            lib.attack_lp.__file__,
            getattr(lib, f"attack_{norm}").__file__,
            lib.attack_utils.__file__,
            lib.utils.__file__,
        ],
    )

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

    # attacks
    attack_loop_kwargs = {
        kwarg.replace("attack_loop_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_loop_")
    }
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS)
        if kwarg.startswith("attack_") and not kwarg.startswith("attack_loop_")
        and kwarg not in ["attack_save"]
    }
    alp = attack_klass(lambda x: test_classifier(x)["logits"], **attack_kwargs)
    alp.build([X_shape, y_shape])
    allp = AttackOptimizationLoop(alp, **attack_loop_kwargs)

    # test metrics
    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        outs = test_classifier(image)
        is_corr = outs["pred"] == label

        label_onehot = tf.one_hot(label, num_classes)
        image_adv = allp.run_loop(image, label_onehot)

        outs_adv = test_classifier(image_adv)
        is_adv = outs_adv["pred"] != label

        # metrics
        nll_loss = tf.keras.metrics.sparse_categorical_crossentropy(
            label, outs["logits"])
        acc_fn = tf.keras.metrics.sparse_categorical_accuracy
        acc = acc_fn(label, outs["logits"])
        acc_adv = acc_fn(label, outs_adv["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_{norm}"](acc_adv)
        test_metrics[f"conf_{norm}"](outs_adv["conf"])

        # measure norm
        r = image - image_adv
        lp = alp.lp_metric(r)
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
        for threshold in test_thresholds[norm]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            test_metrics[f"acc_{norm}_%s" %
                         format_float(threshold, 4)](~is_adv_at_th)
            if norm == "l0":
                is_adv_at_th = tf.logical_and(l0p <= threshold, is_adv)
                test_metrics["acc_l0p_%s" %
                             format_float(threshold, 4)](~is_adv_at_th)
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        is_completed = False
        X_adv = []
        for batch_index, (image, label) in enumerate(test_ds, 1):
            X_adv_b = test_step(image, label)
            X_adv.append(X_adv_b)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                is_completed = True
                break
        else:
            is_completed = True
        X_adv = np.concatenate(X_adv, axis=0)
        if is_completed:
            if FLAGS.attack_save:
                np.save(
                    Path(FLAGS.working_dir) / "attack.npy",
                    X_adv.reshape(X_adv.shape[0], -1))
            # hyperparameter tuning
            with tf.summary.create_file_writer(FLAGS.working_dir).as_default():
                # hyperparameters
                hp_param_names = [
                    kwarg for kwarg in dir(FLAGS)
                    if kwarg.startswith("attack_")
                ]
                hp_metric_names = [f"final_{norm}", f"final_{norm}_corr"]
                hp_params = [
                    hp.HParam(hp_param_name)
                    for hp_param_name in hp_param_names
                ]
                hp_metrics = [
                    hp.Metric(hp_metric_name)
                    for hp_metric_name in hp_metric_names
                ]
                hp.hparams_config(hparams=hp_params, metrics=hp_metrics)
                hp.hparams({
                    hp_param_name: getattr(FLAGS, hp_param_name)
                    for hp_param_name in hp_param_names
                })
                final_lp = test_metrics[f"{norm}"].result()
                tf.summary.scalar(f"final_{norm}", final_lp, step=1)
                final_lp_corr = test_metrics[f"{norm}_corr"].result()
                tf.summary.scalar(f"final_{norm}_corr", final_lp_corr, step=1)
                tf.summary.flush()
    except KeyboardInterrupt as e:
        logging.info("Stopping becaues".format(batch_index))
    except Exception:
        traceback.print_exc()
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                 batch_index),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.attack)
    absl.app.run(main)
