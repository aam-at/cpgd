from __future__ import absolute_import, division, print_function

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
from lib.attack_l2 import OptimizerL2
from lib.utils import (MetricsDictionary, compute_norms,
                       get_acc_for_lp_threshold, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, select_balanced_subset,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_l2")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")
flags.DEFINE_bool("sort_labels", False, "sort labels")

# attack parameters
flags.DEFINE_float("attack_learning_rate", 1e-2, "learning rate for primal variables")
flags.DEFINE_float("attack_lambda_learning_rate", 1e-1, "learning rate for dual variables")
flags.DEFINE_integer("attack_max_iter", 1000, "max iterations")
flags.DEFINE_integer("attack_min_iter_per_start", 0, "min iterations before random restart")
flags.DEFINE_integer("attack_max_iter_per_start", 100, "max iterations before random restart")
flags.DEFINE_float("attack_tol", 0.005, "attack tolerance")
flags.DEFINE_float("attack_round_tol", 0.001, "round attack norm to multiples of")
flags.DEFINE_string("attack_round_op", None, "attack rounding operation")
flags.DEFINE_float("attack_sampling_radius", None, "attack sampling radius")
flags.DEFINE_float("attack_confidence", 0, "margin confidence of adversarial examples")
flags.DEFINE_float("attack_initial_const", 0.1, "initial const for attack")
flags.DEFINE_bool("attack_multitargeted", False, "use multitargeted attack")
flags.DEFINE_bool("attack_proxy_constrain", True, "use proxy for lagrange multiplier maximization")

flags.DEFINE_boolean("generate_summary", False, "generate summary images")
flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in batches)")
flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    setup_experiment("madry_l2_test", [lib.attack_l2.__file__])

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
    classifier = MadryCNN(model_type=model_type)

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    classifier(np.zeros([1, 32, 32, 3], dtype=np.float32))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    # attacks
    ol2 = OptimizerL2(lambda x: test_classifier(x)["logits"],
                      batch_size=FLAGS.batch_size,
                      learning_rate=FLAGS.attack_learning_rate,
                      lambda_learning_rate=FLAGS.attack_lambda_learning_rate,
                      max_iterations=FLAGS.attack_max_iter,
                      min_iterations_per_start=FLAGS.attack_min_iter_per_start,
                      max_iterations_per_start=FLAGS.attack_max_iter_per_start,
                      confidence=FLAGS.attack_confidence,
                      targeted=False,
                      multitargeted=FLAGS.attack_multitargeted,
                      sampling_radius=FLAGS.attack_sampling_radius,
                      tol=FLAGS.attack_tol,
                      round_op=FLAGS.attack_round_op,
                      round_tol=FLAGS.attack_round_tol,
                      initial_const=FLAGS.attack_initial_const,
                      use_proxy_constraint=FLAGS.attack_proxy_constrain)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label, batch_index):
        label_onehot = tf.one_hot(label, num_classes)
        image_l2 = ol2(image, label_onehot)

        outs = test_classifier(image)
        outs_l2 = test_classifier(image_l2)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l2 = acc_fn(label, outs_l2["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l2"](acc_l2)
        test_metrics["conf_l2"](outs_l2["conf"])

        # measure norm
        l2, l2_norm = compute_norms(image, image_l2)
        for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25]:
            acc_th = get_acc_for_lp_threshold(
                lambda x: test_classifier(x)['logits'], image, image_l2, label,
                l2, threshold)
            test_metrics["acc_l2_%.2f" % threshold](acc_th)
        test_metrics["l2"](l2)
        test_metrics["l2_norm"](l2_norm)
        # exclude incorrectly classified
        is_corr = outs['pred'] == label
        test_metrics["l2_corr"](l2[is_corr])

        return image_l2

    if FLAGS.generate_summary:
        start_time = time.time()
        logging.info("Generating samples...")
        summary_images, summary_labels = select_balanced_subset(
            x_test, y_test, num_classes, num_classes)
        summary_images = tf.convert_to_tensor(summary_images)
        summary_labels = tf.convert_to_tensor(summary_labels)
        summary_l2_imgs = test_step(summary_images, summary_labels, -1)
        save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
        save_images(summary_images.numpy(), save_path, data_format="NHWC")
        save_path = os.path.join(FLAGS.samples_dir, 'l2.png')
        save_images(summary_l2_imgs.numpy(), save_path, data_format="NHWC")
        log_metrics(
            test_metrics,
            "Summary results [{:.2f}s]:".format(time.time() - start_time))
    else:
        logging.debug("Skipping summary...")

    # reset metrics
    reset_metrics(test_metrics)
    X_l2_list = []
    y_list = []
    indx_list = []
    start_time = time.time()
    try:
        is_completed = False
        for batch_index, (image, label, indx) in enumerate(test_ds, 1):
            X_l2 = test_step(image, label, batch_index)
            save_path = os.path.join(FLAGS.samples_dir,
                                     'epoch_orig-%d.png' % batch_index)
            save_images(image, save_path, data_format="NHWC")
            save_path = os.path.join(FLAGS.samples_dir,
                                     'epoch_l2-%d.png' % batch_index)
            save_images(X_l2, save_path, data_format="NHWC")
            # save adversarial data
            X_l2_list.append(X_l2)
            y_list.append(label)
            indx_list.append(indx)
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
                    'attack_max_iter', 'attack_tol', 'attack_learning_rate',
                    'attack_lambda_learning_rate', 'attack_initial_const'
                ]
                hp_metric_names = ['final_l2', 'final_l2_corr']
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
                final_l2 = test_metrics['l2'].result()
                tf.summary.scalar('final_l2', final_l2, step=1)
                final_l2_corr = test_metrics['l2_corr'].result()
                tf.summary.scalar('final_l2_corr', final_l2_corr, step=1)
                tf.summary.flush()
    except:
        logging.info("Stopping after {}".format(batch_index))
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                 batch_index))
        X_l2_all = tf.concat(X_l2_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        indx_list = tf.concat(indx_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / 'X_adv',
                 X_adv=X_l2_all,
                 y=y_all,
                 indices=indx_list)


if __name__ == "__main__":
    absl.app.run(main)
