from __future__ import absolute_import, division, print_function

import os
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags

from data import load_cifar10
from lib.attack import DeepFool, FastGradientMethod, HighConfidenceAttack
from lib.utils import (MetricsDictionary, compute_norms,
                       get_acc_for_lp_threshold, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       save_images, setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")
flags.DEFINE_bool("sort_labels", False, "sort labels")

## attack parameters
# FGSM
flags.DEFINE_float("at_epsilon1", 8.0 / 255.0, "fast gradient epsilon")
flags.DEFINE_float("at_epsilon2", 0.1, "fast gradient epsilon")

# High confidence and deepfool
flags.DEFINE_integer("attack_iter", 1000, "maximum number iterations for the attacks")
flags.DEFINE_float("attack_clip", 0.5, "perturbation clip during search")
flags.DEFINE_float("attack_overshoot", 0.02, "multiplier for final perturbation")

flags.DEFINE_integer("summary_frequency", 1, "summarize frequency (in batches)")
flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    setup_experiment(Path(FLAGS.load_from).name)

    # data
    _, _, test_ds = load_cifar10(FLAGS.validation_size)
    x_test, y_test = test_ds._tensors
    x_test, y_test = x_test.numpy(), y_test.numpy()
    indices = np.arange(x_test.shape[0])
    if FLAGS.sort_labels:
        ys_indices = np.argsort(y_test)
        x_test = x_test[ys_indices]
        y_test = y_test[ys_indices]

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

    # compute score
    @tf.function
    def compute_score(image, label=None, in_dist=True):
        logits = test_classifier(image)["logits"]
        score = tf.reduce_max(tf.nn.softmax(logits), axis=-1)
        if in_dist:
            assert label is not None
            pred = tf.argmax(logits, axis=-1)
            return score, score[pred == label], score[pred != label]
        else:
            return score

    in_scores, right_scores, wrong_scores = [], [], []
    # out of distribution detection
    for image_in, label, _ in test_ds:
        in_score, right_score, wrong_score = compute_score(image_in,
                                                           label,
                                                           in_dist=True)
        in_scores.append(in_score)
        right_scores.append(right_score)
        wrong_scores.append(wrong_score)
    in_scores = np.hstack(in_scores)
    right_scores = np.hstack(right_scores)
    wrong_scores = np.hstack(wrong_scores)
    conf_0_95 = -np.cast[np.float32](np.quantile(-in_scores, 0.95))

    # attacks
    deepfool = DeepFool(
        lambda x: test_classifier(x)["logits"],
        max_iter=FLAGS.attack_iter,
        over_shoot=FLAGS.attack_overshoot,
        clip_dist=FLAGS.attack_clip,
        boxmin=0.0,
        boxmax=1.0,
    )
    hc = HighConfidenceAttack(
        lambda x: classifier(x, training=False)["logits"],
        confidence=conf_0_95,
        max_iter=FLAGS.attack_iter,
        over_shoot=FLAGS.attack_overshoot,
        clip_dist=FLAGS.attack_clip,
        attack_random=False,
        attack_uniform=False,
        boxmin=0.0,
        boxmax=1.0,
    )
    at1 = FastGradientMethod(
        lambda x: test_classifier(x)["logits"],
        epsilon=FLAGS.at_epsilon1,
        boxmin=0.0,
        boxmax=1.0,
    )
    at2 = FastGradientMethod(
        lambda x: test_classifier(x)["logits"],
        epsilon=FLAGS.at_epsilon2,
        boxmin=0.0,
        boxmax=1.0,
    )

    test_metrics = MetricsDictionary({
        "nll_loss":
        tf.keras.metrics.SparseCategoricalCrossentropy(name="nll_loss",
                                                       from_logits=True),
        "acc":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        "acc_at1":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc_at1"),
        "acc_at2":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc_at2"),
        "acc_df":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc_df"),
        "acc_hc":
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc_hc"),
    })

    @tf.function
    def test_step(image, label):
        image_df = deepfool(image, label)
        image_hc = hc(image, label)
        image_at1 = at1(image, label)
        image_at2 = at2(image, label)

        # measure norm
        l2_df, l2_df_norm = compute_norms(image, image_df)
        for threshold in [
                0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25
        ]:
            acc_th = get_acc_for_lp_threshold(
                lambda x: test_classifier(x)['logits'],
                image,
                image_df,
                label,
                l2_df,
                threshold=threshold)
            test_metrics["acc_l2_%.2f" % threshold](acc_th)

        outs = test_classifier(image)
        outs_df = test_classifier(image_df)
        outs_hc = test_classifier(image_hc)
        outs_at1 = test_classifier(image_at1)
        outs_at2 = test_classifier(image_at2)

        test_metrics["nll_loss"](label, outs["logits"])
        test_metrics["acc"](label, outs["logits"])
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_at1"](label, outs_at1["logits"])
        test_metrics["conf_at1"](outs_at1["conf"])
        test_metrics["acc_at2"](label, outs_at2["logits"])
        test_metrics["conf_at2"](outs_at2["conf"])
        test_metrics["acc_df"](label, outs_df["logits"])
        test_metrics["conf_df"](outs_df["conf"])
        test_metrics["acc_hc"](label, outs_hc["logits"])
        test_metrics["conf_hc"](outs_hc["conf"])
        # compute psnr and ssim
        psnr_df = tf.image.psnr(
            tf.transpose(image, (0, 2, 3, 1)),
            tf.transpose(image_df, (0, 2, 3, 1)),
            max_val=1.0,
        )
        test_metrics["psnr_df"](psnr_df[~tf.math.is_nan(psnr_df)])
        test_metrics["ssim_df"](tf.image.ssim(image, image_df, max_val=1.0))

        test_metrics["l2_df"](l2_df)
        test_metrics["l2_df_norm"](l2_df_norm)

        l2_hc, l2_hc_norm = compute_norms(image, image_hc)
        test_metrics["l2_hc"](l2_hc)
        test_metrics["l2_hc_norm"](l2_hc_norm)

        return image_df, image_hc, image_at1, image_at2

    summary_writer = tf.summary.create_file_writer(FLAGS.working_dir)
    summary_writer.set_as_default()

    start_time = time.time()
    for batch_index, (image, label, indx) in enumerate(test_ds, 1):
        X_df, X_hc, X_at1, X_at2 = test_step(image, label)
        save_path = os.path.join(FLAGS.samples_dir,
                                 "epoch_orig-%d.png" % batch_index)
        save_images(image, save_path, data_format="NHWC")
        save_path = os.path.join(FLAGS.samples_dir,
                                 "epoch_df-%d.png" % batch_index)
        save_images(X_df, save_path, data_format="NHWC")
        save_path = os.path.join(FLAGS.samples_dir,
                                 "epoch_hc-%d.png" % batch_index)
        save_images(X_hc, save_path, data_format="NHWC")
        save_path = os.path.join(FLAGS.samples_dir,
                                 "epoch_at1-%d.png" % batch_index)
        save_images(X_at1, save_path, data_format="NHWC")
        save_path = os.path.join(FLAGS.samples_dir,
                                 "epoch_at2-%d.png" % batch_index)
        save_images(X_at2, save_path, data_format="NHWC")
        if batch_index % FLAGS.print_frequency == 0:
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
        if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
            break
    log_metrics(test_metrics,
                "Test results [{:.2f}s]:".format(time.time() - start_time))


if __name__ == "__main__":
    absl.app.run(main)
