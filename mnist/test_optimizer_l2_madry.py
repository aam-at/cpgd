from __future__ import absolute_import, division, print_function

import logging
import os
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags

import lib
from data import load_mnist
from lib.attack_l2 import OptimizerL2
from lib.utils import (MetricsDictionary, compute_norms, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, select_balanced_subset,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_l2")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")
flags.DEFINE_bool("sort_labels", False, "sort labels")

# attack parameters
flags.DEFINE_integer("attack_max_iter", 10000, "max iterations")
flags.DEFINE_integer("attack_min_restart_iter", 10, "min iterations before random restart")
flags.DEFINE_integer("attack_max_restart_iter", 100, "max iterations before random restart")
flags.DEFINE_string("attack_r0_init", "normal", "r0 initializer")
flags.DEFINE_float("attack_tol", 5e-3, "attack tolerance")
flags.DEFINE_float("attack_confidence", 0, "margin confidence of adversarial examples")
flags.DEFINE_float("attack_initial_const", 1e2, "initial const for attack")
flags.DEFINE_bool("attack_multitargeted", False, "use multitargeted attack")
flags.DEFINE_bool("attack_proxy_constrain", True, "use proxy for lagrange multiplier maximization")

flags.DEFINE_boolean("generate_summary", False, "generate summary images")
flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in batches)")
flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment("madry_l2_test", [lib.attack_l2.__file__])

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size, seed=FLAGS.data_seed)
    x_test, y_test = test_ds._tensors
    x_test, y_test = x_test.numpy(), y_test.numpy()
    x_test = x_test.transpose(0, 2, 3, 1)
    indices = np.arange(x_test.shape[0])
    if FLAGS.sort_labels:
        ys_indices = np.argsort(y_test)
        x_test = x_test[ys_indices]
        y_test = y_test[ys_indices]
        indices = indices[ys_indices]

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test, indices))
    test_ds = make_input_pipeline(test_ds, shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    classifier = MadryCNN()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    classifier(np.zeros([1, 28, 28, 1], dtype=np.float32))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # attacks
    ol2 = OptimizerL2(lambda x: test_classifier(x)["logits"],
                      batch_size=FLAGS.batch_size,
                      confidence=FLAGS.attack_confidence,
                      targeted=False,
                      multitargeted=FLAGS.attack_multitargeted,
                      r0_init=FLAGS.attack_r0_init,
                      max_iterations=FLAGS.attack_max_iter,
                      min_restart_iterations=FLAGS.attack_min_restart_iter,
                      max_restart_iterations=FLAGS.attack_max_restart_iter,
                      tol=FLAGS.attack_tol,
                      initial_const=FLAGS.attack_initial_const,
                      use_proxy_constraint=FLAGS.attack_proxy_constrain)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label, batch_index):
        label_onehot = tf.one_hot(label, num_classes)
        image_l2 = ol2(image, label_onehot)
        # measure norm
        l2, l2_norm = compute_norms(image, image_l2)

        image_l2_0_5 = tf.where(tf.reshape(l2 <= 0.5, (-1, 1, 1, 1)), image_l2, image)
        image_l2_1 = tf.where(tf.reshape(l2 <= 1.0, (-1, 1, 1, 1)), image_l2, image)
        image_l2_1_5 = tf.where(tf.reshape(l2 <= 1.5, (-1, 1, 1, 1)), image_l2, image)
        image_l2_2_0 = tf.where(tf.reshape(l2 <= 2.0, (-1, 1, 1, 1)), image_l2, image)
        image_l2_2_5 = tf.where(tf.reshape(l2 <= 2.5, (-1, 1, 1, 1)), image_l2, image)
        image_l2_3_0 = tf.where(tf.reshape(l2 <= 3.0, (-1, 1, 1, 1)), image_l2, image)

        outs = test_classifier(image)
        outs_l2 = test_classifier(image_l2)
        outs_l2_0_5 = test_classifier(image_l2_0_5)
        outs_l2_1 = test_classifier(image_l2_1)
        outs_l2_1_5 = test_classifier(image_l2_1_5)
        outs_l2_2_0 = test_classifier(image_l2_2_0)
        outs_l2_2_5 = test_classifier(image_l2_2_5)
        outs_l2_3_0 = test_classifier(image_l2_3_0)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l2 = acc_fn(label, outs_l2["logits"])
        acc_l2_0_5 = acc_fn(label, outs_l2_0_5["logits"])
        acc_l2_1 = acc_fn(label, outs_l2_1["logits"])
        acc_l2_1_5 = acc_fn(label, outs_l2_1_5["logits"])
        acc_l2_2_0 = acc_fn(label, outs_l2_2_0["logits"])
        acc_l2_2_5 = acc_fn(label, outs_l2_2_5["logits"])
        acc_l2_3_0 = acc_fn(label, outs_l2_3_0["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l2"](acc_l2)
        test_metrics["conf_l2"](outs_l2["conf"])
        test_metrics["acc_l2_0.5"](acc_l2_0_5)
        test_metrics["acc_l2_1.0"](acc_l2_1)
        test_metrics["acc_l2_1.5"](acc_l2_1_5)
        test_metrics["acc_l2_2.0"](acc_l2_2_0)
        test_metrics["acc_l2_2.5"](acc_l2_2_5)
        test_metrics["acc_l2_3.0"](acc_l2_3_0)

        test_metrics["l2"](l2)
        test_metrics["l2_norm"](l2_norm)
        # exclude incorrectly classified
        is_corr = outs['pred'] == label
        test_metrics["l2_corr"](l2[is_corr])

        # summaries
        tf.summary.scalar("l2", tf.reduce_mean(l2), batch_index)
        tf.summary.scalar("l2_norm", tf.reduce_mean(l2_norm), batch_index)

        return image_l2

    summary_writer = tf.summary.create_file_writer(FLAGS.working_dir)
    summary_writer.set_as_default()

    if FLAGS.generate_summary:
        start_time = time.time()
        logging.info("Generating samples...")
        summary_images, summary_labels = select_balanced_subset(
            x_test, y_test, num_classes, num_classes)
        summary_images = tf.convert_to_tensor(summary_images)
        summary_labels = tf.convert_to_tensor(summary_labels)
        summary_l2_imgs = test_step(summary_images, summary_labels, -1)
        save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
        save_images(summary_images.numpy(), save_path)
        save_path = os.path.join(FLAGS.samples_dir, 'l2.png')
        save_images(summary_l2_imgs.numpy(), save_path)
        log_metrics(
            test_metrics,
            "Summary results [{:.2f}s]:".format(time.time() - start_time))
    else:
        logging.debug("Skipping summary...")

    reset_metrics(test_metrics)
    X_l2_list = []
    y_list = []
    indx_list = []
    start_time = time.time()
    try:
        for batch_index, (image, label, indx) in enumerate(test_ds, 1):
            X_l2 = test_step(image, label, batch_index)
            image = np.transpose(image, (0, 3, 1, 2))
            X_l2 = np.transpose(X_l2, (0, 3, 1, 2))
            save_path = os.path.join(FLAGS.samples_dir,
                                     'epoch_orig-%d.png' % batch_index)
            save_images(image, save_path)
            save_path = os.path.join(FLAGS.samples_dir,
                                     'epoch_l2-%d.png' % batch_index)
            save_images(X_l2, save_path)
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
                break
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
