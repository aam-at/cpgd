from __future__ import absolute_import, division, print_function

import logging
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from tensorboard.plugins.hparams import api as hp

from data import load_mnist
from foolbox.attacks import EADAttack
from foolbox.models import TensorFlowModel
from lib.utils import (MetricsDictionary, get_acc_for_lp_threshold,
                       import_kwargs_as_flags, l1_metric, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_brendel_lp")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")

# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
import_kwargs_as_flags(EADAttack.__init__, "attack_")
flags.DEFINE_string("attack_decision_rule", "L1", "attack decision rule")

flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment("madry_ead_l1_test")

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    classifier = MadryCNN()
    fclassifier = TensorFlowModel(lambda x: classifier(x)["logits"],
                                  bounds=(0.0, 1.0))

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    classifier(np.zeros([1, 28, 28, 1], dtype=np.float32))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # attacks
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
        and kwarg not in ["attack_random_restarts"]
    }
    olp = EADAttack(**attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    test_thresholds = ([
        2.0, 2.5, 4.0, 5.0, 6.0, 7.5, 8.0, 8.75, 10.0, 12.5, 16.25, 20.0
    ], )

    def test_step(image, label):
        image_lp = olp(fclassifier, image, label, epsilons=None)

        outs = test_classifier(image)
        outs_l1 = test_classifier(image_lp)

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

        l1 = l1_metric(image - image_lp)
        for threshold in test_thresholds[FLAGS.norm]:
            acc_th = get_acc_for_lp_threshold(
                lambda x: test_classifier(x)["logits"],
                image,
                image_lp,
                label,
                l1,
                threshold,
            )
            test_metrics["acc_l1_%.2f" % threshold](acc_th)
        test_metrics["l1"](l1)

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
    start_time = time.time()
    try:
        is_completed = False
        for batch_index, (image, label) in enumerate(test_ds, 1):
            test_step(image, label)
            if batch_index % FLAGS.print_frequency == 0:
                log_metrics(
                    test_metrics,
                    "Batch results [{}, {:.2f}s]:".format(
                        batch_index,
                        time.time() - start_time),
                )
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                is_completed = True
                break
        else:
            is_completed = True
        if is_completed:
            # hyperparameter tuning
            with tf.summary.create_file_writer(FLAGS.working_dir).as_default():
                # hyperparameters
                hp_param_names = [
                    kwarg for kwarg in dir(FLAGS)
                    if kwarg.startswith("attack_")
                ]
                hp_metric_names = [
                    f"final_{FLAGS.norm}", f"final_{FLAGS.norm}_corr"
                ]
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
                final_lp = test_metrics[f"{FLAGS.norm}"].result()
                tf.summary.scalar(f"final_{FLAGS.norm}", final_lp, step=1)
                final_lp_corr = test_metrics[f"{FLAGS.norm}_corr"].result()
                tf.summary.scalar(f"final_{FLAGS.norm}_corr",
                                  final_lp_corr,
                                  step=1)
                tf.summary.flush()
    except:
        logging.info("Stopping after {}".format(batch_index))
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                 batch_index),
        )
        X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / "X_adv", X_adv=X_lp_all, y=y_all)


if __name__ == "__main__":
    absl.app.run(main)
