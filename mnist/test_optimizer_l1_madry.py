from __future__ import absolute_import, division, print_function

import logging
import os
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags

from attack_l1 import OptimizerL1
from data import load_mnist, make_input_pipeline, select_balanced_subset
from models import MadryCNN
from utils import (MetricsDictionary, l1_metric, log_metrics,
                   register_experiment_flags, reset_metrics, save_images,
                   setup_experiment)

# general experiment parameters
register_experiment_flags(working_dir="test_l1")
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
flags.DEFINE_bool("attack_proxy_constrain", True, "use proxy for lagrange multiplier maximization")

flags.DEFINE_boolean("generate_summary", False, "generate summary images")
flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in batches)")
flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS


def load_madry(load_dir, model_vars):
    import scipy.io
    w = scipy.io.loadmat(load_dir)
    mapping = {
        "A0": "conv2d/kernel:0",
        "A1": "conv2d/bias:0",
        "A2": "conv2d_1/kernel:0",
        "A3": "conv2d_1/bias:0",
        "A4": "dense/kernel:0",
        "A5": "dense/bias:0",
        "A6": "dense_1/kernel:0",
        "A7": "dense_1/bias:0"
    }
    for var_name in w.keys():
        if not var_name.startswith("A"):
            continue
        var = w[var_name]
        if var.ndim == 2:
            var = var.squeeze()
        model_var_name = mapping[var_name]
        model_var = [
            v for v in model_vars if v.name == model_var_name
        ]
        assert len(model_var) == 1
        model_var[0].assign(var)


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    setup_experiment("madry_l1_test", "attack_l1.py")

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
    ol1 = OptimizerL1(lambda x: test_classifier(x)["logits"],
                      batch_size=FLAGS.batch_size,
                      confidence=FLAGS.attack_confidence,
                      targeted=False,
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
        image_l1 = ol1(image, label_onehot)
        # measure norm
        l1 = l1_metric(image - image_l1)

        image_l1_2 = tf.where(tf.reshape(l1 <= 2.0, (-1, 1, 1, 1)), image_l1, image)
        image_l1_2_5 = tf.where(tf.reshape(l1 <= 2.5, (-1, 1, 1, 1)), image_l1, image)
        image_l1_4 = tf.where(tf.reshape(l1 <= 4.0, (-1, 1, 1, 1)), image_l1, image)
        image_l1_5 = tf.where(tf.reshape(l1 <= 5.0, (-1, 1, 1, 1)), image_l1, image)
        image_l1_6 = tf.where(tf.reshape(l1 <= 6.0, (-1, 1, 1, 1)), image_l1, image)
        image_l1_7_5 = tf.where(tf.reshape(l1 <= 7.5, (-1, 1, 1, 1)), image_l1, image)
        image_l1_8 = tf.where(tf.reshape(l1 <= 8.0, (-1, 1, 1, 1)), image_l1, image)
        image_l1_8_75 = tf.where(tf.reshape(l1 <= 8.75, (-1, 1, 1, 1)), image_l1, image)
        image_l1_10_0 = tf.where(tf.reshape(l1 <= 10.0, (-1, 1, 1, 1)), image_l1, image)
        image_l1_12_5 = tf.where(tf.reshape(l1 <= 12.5, (-1, 1, 1, 1)), image_l1, image)
        image_l1_16_25 = tf.where(tf.reshape(l1 <= 16.25, (-1, 1, 1, 1)), image_l1, image)
        image_l1_20 = tf.where(tf.reshape(l1 <= 20.0, (-1, 1, 1, 1)), image_l1, image)

        outs = test_classifier(image)
        outs_l1 = test_classifier(image_l1)
        outs_l1_2 = test_classifier(image_l1_2)
        outs_l1_2_5 = test_classifier(image_l1_2_5)
        outs_l1_4 = test_classifier(image_l1_4)
        outs_l1_5 = test_classifier(image_l1_5)
        outs_l1_6 = test_classifier(image_l1_6)
        outs_l1_7_5 = test_classifier(image_l1_7_5)
        outs_l1_8 = test_classifier(image_l1_8)
        outs_l1_8_75 = test_classifier(image_l1_8_75)
        outs_l1_10_0 = test_classifier(image_l1_10_0)
        outs_l1_12_5 = test_classifier(image_l1_12_5)
        outs_l1_16_25 = test_classifier(image_l1_16_25)
        outs_l1_20_0 = test_classifier(image_l1_20)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l1 = acc_fn(label, outs_l1["logits"])
        acc_l1_2 = acc_fn(label, outs_l1_2["logits"])
        acc_l1_2_5 = acc_fn(label, outs_l1_2_5["logits"])
        acc_l1_4 = acc_fn(label, outs_l1_4["logits"])
        acc_l1_5 = acc_fn(label, outs_l1_5["logits"])
        acc_l1_6 = acc_fn(label, outs_l1_6["logits"])
        acc_l1_7_5 = acc_fn(label, outs_l1_7_5["logits"])
        acc_l1_8 = acc_fn(label, outs_l1_8["logits"])
        acc_l1_8_75 = acc_fn(label, outs_l1_8_75["logits"])
        acc_l1_10_0 = acc_fn(label, outs_l1_10_0["logits"])
        acc_l1_12_5 = acc_fn(label, outs_l1_12_5["logits"])
        acc_l1_16_25 = acc_fn(label, outs_l1_16_25["logits"])
        acc_l1_20_0 = acc_fn(label, outs_l1_20_0["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l1"](acc_l1)
        test_metrics["conf_l2"](outs_l1["conf"])
        test_metrics["acc_l1_2.0"](acc_l1_2)
        test_metrics["acc_l1_2.5"](acc_l1_2_5)
        test_metrics["acc_l1_4.0"](acc_l1_4)
        test_metrics["acc_l1_5.0"](acc_l1_5)
        test_metrics["acc_l1_6.0"](acc_l1_6)
        test_metrics["acc_l1_7.5"](acc_l1_7_5)
        test_metrics["acc_l1_8.0"](acc_l1_8)
        test_metrics["acc_l1_8.75"](acc_l1_8_75)
        test_metrics["acc_l1_10.0"](acc_l1_10_0)
        test_metrics["acc_l1_12.5"](acc_l1_12_5)
        test_metrics["acc_l1_16.25"](acc_l1_16_25)
        test_metrics["acc_l1_20.0"](acc_l1_20_0)

        test_metrics["l1"](l1)
        # exclude incorrectly classified
        is_corr = outs['pred'] == label
        test_metrics["l1_corr"](l1[is_corr])

        # summaries
        tf.summary.scalar("l1", tf.reduce_mean(l1), batch_index)

        return image_l1

    summary_writer = tf.summary.create_file_writer(FLAGS.working_dir)
    summary_writer.set_as_default()

    if FLAGS.generate_summary:
        start_time = time.time()
        logging.info("Generating samples...")
        summary_images, summary_labels = select_balanced_subset(
            x_test, y_test, num_classes, num_classes)
        summary_images = tf.convert_to_tensor(summary_images)
        summary_labels = tf.convert_to_tensor(summary_labels)
        summary_l1_imgs = test_step(summary_images, summary_labels, -1)
        save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
        save_images(summary_images.numpy(), save_path)
        save_path = os.path.join(FLAGS.samples_dir, 'l1.png')
        save_images(summary_l1_imgs.numpy(), save_path)
        log_metrics(
            test_metrics,
            "Summary results [{:.2f}s]:".format(time.time() - start_time))
    else:
        logging.debug("Skipping summary...")

    reset_metrics(test_metrics)
    X_l1_list = []
    y_list = []
    indx_list = []
    start_time = time.time()
    try:
        for batch_index, (image, label, indx) in enumerate(test_ds, 1):
            X_l1 = test_step(image, label, batch_index)
            image = np.transpose(image, (0, 3, 1, 2))
            X_l1 = np.transpose(X_l1, (0, 3, 1, 2))
            save_path = os.path.join(FLAGS.samples_dir,
                                     'epoch_orig-%d.png' % batch_index)
            save_images(image, save_path)
            save_path = os.path.join(FLAGS.samples_dir,
                                     'epoch_l1-%d.png' % batch_index)
            save_images(X_l1, save_path)
            # save adversarial data
            X_l1_list.append(X_l1)
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
        X_l1_all = tf.concat(X_l1_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        indx_list = tf.concat(indx_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / 'X_adv',
                 X_adv=X_l1_all,
                 y=y_all,
                 indices=indx_list)


if __name__ == "__main__":
    absl.app.run(main)
