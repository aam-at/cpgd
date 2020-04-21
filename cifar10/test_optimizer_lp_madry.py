from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from tensorboard.plugins.hparams import api as hp

import lib
from data import load_cifar10
from lib.attack_l0 import OptimizerL0
from lib.attack_l1 import OptimizerL1
from lib.attack_l2 import OptimizerL2
from lib.attack_li import OptimizerLi
from lib.attack_lp import OptimizerLp
from lib.utils import (MetricsDictionary, get_acc_for_lp_threshold,
                       import_kwargs_as_flags, l0_metric, l1_metric, l2_metric,
                       li_metric, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_lp")
flags.DEFINE_string("norm", "l1", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
import_kwargs_as_flags(OptimizerLp.__init__, 'attack_')

flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.norm in ['l0', 'l1', 'l2', 'li']
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_{FLAGS.norm}_test", [
        lib.attack_lp.__file__,
        getattr(lib, f"attack_{FLAGS.norm}").__file__
    ])

    # data
    _, _, test_ds = load_cifar10(FLAGS.validation_size)
    x_test, y_test = test_ds._tensors
    x_test, y_test = x_test.numpy(), y_test.numpy()
    indices = np.arange(x_test.shape[0])
    if FLAGS.sort_labels:
        ys_indices = np.argsort(y_test)
        x_test = x_test[ys_indices]
        y_test = y_test[ys_indices]
        indices = indices[ys_indices]

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test, indices))
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    classifier = MadryCNN(model_type=model_type)

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    classifier(np.zeros([1, 32, 32, 3], dtype=np.float32))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    lp_attacks = {
        'l0': OptimizerL0,
        'l1': OptimizerL1,
        'l2': OptimizerL2,
        'li': OptimizerLi
    }
    lp_metrics = {
        'l0': l0_metric,
        'l1': l1_metric,
        'l2': l2_metric,
        'li': li_metric
    }
    test_thresholds = {
        'l0': [20, 40, 60, 80, 100],
        'l1': [
            2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 8.75, 9.0, 10.0, 12.0, 12.5, 15.0,
            16.25, 20.0
        ],
        'l2': [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25],
        'li': [
            1 / 255, 1.5 / 255, 2 / 255, 2.5 / 255, 3 / 255, 4 / 255, 6 / 255,
            8 / 255, 10 / 255
        ]
    }
    # attacks
    attack_kwargs = {
        kwarg.replace('attack_', ''): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith('attack_')
    }
    olp = lp_attacks[FLAGS.norm](lambda x: test_classifier(x)["logits"],
                                 batch_size=FLAGS.batch_size,
                                 **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        image_lp = olp(image, label_onehot)

        outs = test_classifier(image)
        outs_lp = test_classifier(image_lp)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_lp = acc_fn(label, outs_lp["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_{FLAGS.norm}"](acc_lp)
        test_metrics[f"conf_{FLAGS.norm}"](outs_lp["conf"])

        # measure norm
        lp = lp_metrics[FLAGS.norm](image - image_lp)
        for threshold in test_thresholds[FLAGS.norm]:
            acc_th = get_acc_for_lp_threshold(
                lambda x: test_classifier(x)['logits'], image, image_lp, label,
                lp, threshold)
            test_metrics[f"acc_{FLAGS.norm}_%.2f" % threshold](acc_th)
        test_metrics[f"{FLAGS.norm}"](lp)
        # exclude incorrectly classified
        is_corr = outs['pred'] == label
        test_metrics[f"{FLAGS.norm}_corr"](lp[is_corr])

        return image_lp

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
    start_time = time.time()
    try:
        is_completed = False
        for batch_index, (image, label) in enumerate(test_ds, 1):
            X_lp = test_step(image, label)
            save_path = os.path.join(FLAGS.samples_dir,
                                     "epoch_orig-%d.png" % batch_index)
            save_images(image, save_path, data_format="NHWC")
            save_path = os.path.join(
                FLAGS.samples_dir, f"epoch_{FLAGS.norm}-%d.png" % batch_index)
            save_images(X_lp, save_path, data_format="NHWC")
            # save adversarial data
            X_lp_list.append(X_lp)
            y_list.append(label)
            if batch_index % FLAGS.print_frequency == 0:
                log_metrics(
                    test_metrics, "Batch results [{}, {:.2f}s]:".format(
                        batch_index,
                        time.time() - start_time))
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
                    if kwarg.startswith('attack_')
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
                                                 batch_index))
        X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / 'X_adv', X_adv=X_lp_all, y=y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    if args.norm == 'li':
        import_kwargs_as_flags(OptimizerLi.__init__, 'attack_')
    absl.app.run(main)
