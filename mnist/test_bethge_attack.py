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

from data import load_mnist
from foolbox.attacks import (L0BrendelBethgeAttack, L1BrendelBethgeAttack,
                             L2BrendelBethgeAttack,
                             LinearSearchBlendedUniformNoiseAttack,
                             LinfinityBrendelBethgeAttack, LinfPGD)
from foolbox.models import TensorFlowModel
from lib.utils import (MetricsDictionary, get_acc_for_lp_threshold,
                       import_kwargs_as_flags, l0_metric, l1_metric, l2_metric,
                       li_metric, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_brendel_lp")
flags.DEFINE_string("norm", "l1", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters

flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS

lp_attacks = {
    'l0': L0BrendelBethgeAttack,
    'l1': L1BrendelBethgeAttack,
    'l2': L2BrendelBethgeAttack,
    'li': LinfinityBrendelBethgeAttack
}


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_bethge_{FLAGS.norm}_test")

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
    fclassifier = TensorFlowModel(lambda x: classifier(x)['logits'], bounds=np.array((0.0, 1.0), dtype=np.float64))

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    classifier(np.zeros([1, 28, 28, 1], dtype=np.float32))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    lp_metrics = {
        'l0': l0_metric,
        'l1': l1_metric,
        'l2': l2_metric,
        'li': li_metric
    }
    test_thresholds = {
        'l0': [10, 30, 50, 80, 100],
        'l1':
            [2.0, 2.5, 4.0, 5.0, 6.0, 7.5, 8.0, 8.75, 10.0, 12.5, 16.25, 20.0],
        'l2': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        'li':
            [0.03, 0.05, 0.07, 0.09, 0.1, 0.11, 0.15, 0.2, 0.25, 0.3, 0.325, 0.35]
    }
    # attacks
    attack_kwargs = {
        kwarg.replace('attack_', ''): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith('attack_')
    }
    olp = lp_attacks[FLAGS.norm](init_attack=LinearSearchBlendedUniformNoiseAttack(), **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        image_lp, _, _ = olp(fclassifier, image, label, epsilons=None)

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
    assert args.norm in ['l0', 'l1', 'l2', 'li']
    import_kwargs_as_flags(lp_attacks[args.norm].__init__, 'attack_')
    absl.app.run(main)
