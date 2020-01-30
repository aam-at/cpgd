from __future__ import absolute_import, division, print_function

import logging
import os
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags

from data import load_mnist, make_input_pipeline, select_balanced_subset
from models import create_model, register_model_flags
from utils import (compute_norms, load_experiment, log_metrics,
                   register_experiment_flags, reset_metrics, save_images,
                   setup_experiment)

# general experiment parameters
register_experiment_flags(working_dir="test_ca")
register_model_flags(model="mlp")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")
flags.DEFINE_bool("sort_labels", True, "sort labels")

# attack parameters
flags.DEFINE_integer("carlini_max_iter", 10000, "max iterations (default: 1000)")
flags.DEFINE_integer("carlini_binary_steps", 9, "number of binary steps")
flags.DEFINE_float("carlini_confidence", 0, "margin confidence of adversarial examples")
flags.DEFINE_float("carlini_lb", 0, "lower bound for carlini C")
flags.DEFINE_float("carlini_ub", 1e4, "upper bound for carlini C")
flags.DEFINE_bool("use_carlini_prob", False, "use probability margin calrini")

flags.DEFINE_boolean("generate_summary", True, "generate summary images")
flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in batches)")
flags.DEFINE_integer("print_frequency", 10, "summarize frequency")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    load_experiment(FLAGS.load_from)
    setup_experiment(Path(FLAGS.load_from).name)

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size, shuffle=True, seed=FLAGS.data_seed)
    x_test, y_test = test_ds._tensors
    x_test, y_test = x_test.numpy(), y_test.numpy()
    if FLAGS.sort_labels:
        ys_indices = np.argsort(y_test)
        x_test = x_test[ys_indices]
        y_test = y_test[ys_indices]

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = make_input_pipeline(test_ds, shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    classifier = create_model(num_classes)

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    chk_dir = Path(FLAGS.load_from) / 'chks'
    chk = tf.train.Checkpoint(classifier=classifier)
    status = chk.restore(tf.train.latest_checkpoint(chk_dir))
    # force to load parameters
    classifier(np.zeros([1, 1, 28, 28], dtype=np.float32))

    # compute score
    @tf.function
    def compute_score(image, label=None, in_dist=True):
        logits = test_classifier(image)['logits']
        score = tf.reduce_max(tf.nn.softmax(logits),
                              axis=-1)
        if in_dist:
            assert label is not None
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            return score, score[pred == label], score[pred != label]
        else:
            return score

    in_scores, right_scores, wrong_scores = [], [], []
    # out of distribution detection
    for image_in, label in test_ds:
        in_score, right_score, wrong_score = compute_score(image_in, label,
                                                           in_dist=True)
        in_scores.append(in_score)
        right_scores.append(right_score)
        wrong_scores.append(wrong_score)
    in_scores = np.hstack(in_scores)
    right_scores = np.hstack(right_scores)
    wrong_scores = np.hstack(wrong_scores)
    conf_0_95 = -np.cast[np.float32](np.quantile(-in_scores, 0.95))

    # attacks
    if FLAGS.use_carlini_prob:
        from attack import CWL2Prob as CWL2
        cwl2 = CWL2(lambda x: test_classifier(x)["logits"],
                    batch_size=FLAGS.batch_size,
                    targeted=False,
                    prob_confidence=conf_0_95,
                    max_iterations=FLAGS.carlini_max_iter,
                    binary_search_steps=FLAGS.carlini_binary_steps,
                    lower_bound=FLAGS.carlini_lb,
                    upper_bound=FLAGS.carlini_ub)
    else:
        from attack import CWL2 as CWL2
        cwl2 = CWL2(lambda x: test_classifier(x)["logits"],
                    batch_size=FLAGS.batch_size,
                    targeted=False,
                    confidence=FLAGS.carlini_confidence,
                    max_iterations=FLAGS.carlini_max_iter,
                    binary_search_steps=FLAGS.carlini_binary_steps,
                    lower_bound=FLAGS.carlini_lb,
                    upper_bound=FLAGS.carlini_ub)

    test_metrics = {
        "nll_loss":
        tf.keras.metrics.SparseCategoricalCrossentropy(name="nll_loss",
                                                       from_logits=True),
        "acc":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        "conf":
        tf.keras.metrics.Mean(name="conf"),
        "acc_ca":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc_ca"),
        'conf_ca':
        tf.keras.metrics.Mean(name="conf_ca"),
        'l2_ca':
        tf.keras.metrics.Mean(name="l2_ca"),
        'l2_ca_norm':
        tf.keras.metrics.Mean(name="l2_ca_norm"),
        'psnr_ca':
        tf.keras.metrics.Mean(name="psnr_ca"),
        'ssim_ca':
        tf.keras.metrics.Mean(name="ssim_ca")
    }

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        image_ca = cwl2(image, label_onehot)

        outs = test_classifier(image)
        outs_ca = test_classifier(image_ca)

        test_metrics["nll_loss"](label, outs["logits"])
        test_metrics["acc"](label, outs["logits"])
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_ca"](label, outs_ca["logits"])
        test_metrics["conf_ca"](outs_ca["conf"])
        # compute psnr and ssim
        psnr_ca = tf.image.psnr(
            tf.transpose(image, (0, 2, 3, 1)),
            tf.transpose(image_ca, (0, 2, 3, 1)),
            max_val=1.0)
        test_metrics["psnr_ca"](psnr_ca[~tf.math.is_nan(psnr_ca)])
        test_metrics["ssim_ca"](tf.image.ssim(
            tf.transpose(image, (0, 2, 3, 1)),
            tf.transpose(image_ca, (0, 2, 3, 1)),
            max_val=1.0))

        l2_ca, l2_ca_norm = compute_norms(image, image_ca)
        test_metrics["l2_ca"](l2_ca)
        test_metrics["l2_ca_norm"](l2_ca_norm)

        return image_ca

    summary_writer = tf.summary.create_file_writer(FLAGS.working_dir)
    summary_writer.set_as_default()

    if FLAGS.generate_summary:
        start_time = time.time()
        logging.info("Generating samples...")
        summary_images, summary_labels = select_balanced_subset(
            x_test, y_test, num_classes, num_classes)
        summary_images = tf.convert_to_tensor(summary_images)
        summary_ca_l2_imgs = test_step(summary_images, summary_labels)
        save_path = os.path.join(FLAGS.samples_dir, 'orig.png')
        save_images(summary_images.numpy(), save_path)
        save_path = os.path.join(FLAGS.samples_dir, 'carlini_l2.png')
        save_images(summary_ca_l2_imgs.numpy(), save_path)
        log_metrics(
            test_metrics, "Summary results [{:.2f}s]:".format(
                time.time() - start_time))
    else:
        logging.debug("Skipping summary...")

    reset_metrics(test_metrics)
    start_time = time.time()
    for batch_index, (image, label) in enumerate(test_ds, 1):
        X_ca = test_step(image, label)
        save_path = os.path.join(FLAGS.samples_dir, 'epoch_orig-%d.png' % batch_index)
        save_images(image.numpy(), save_path)
        save_path = os.path.join(FLAGS.samples_dir, 'epoch_ca-%d.png' % batch_index)
        save_images(X_ca.numpy(), save_path)
        if batch_index % FLAGS.print_frequency == 0:
            log_metrics(
                test_metrics, "Batch results [{}, {:.2f}s]:".format(
                    batch_index, time.time() - start_time))
    log_metrics(
        test_metrics, "Test results [{:.2f}s]:".format(
            time.time() - start_time))


if __name__ == "__main__":
    absl.app.run(main)
