from __future__ import absolute_import, division, print_function

import logging
import time

import absl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

from data import load_mnist
from lib.attack_utils import init_r0, project_box
from lib.tf_utils import MetricsDictionary, make_input_pipeline
from lib.utils import (log_metrics, register_experiment_flags, reset_metrics,
                       setup_experiment)
from models import MadryCNNTf
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_random_lp")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("restarts", 100, "number of random restarts")
flags.DEFINE_string("norm", "l2", "norm")
flags.DEFINE_string("init", "uniform", "random initialization")
flags.DEFINE_float("epsilon", 0.1, "random epsilon")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_test")

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
    classifier = MadryCNNTf()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 28, 28, 1])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # test metrics
    test_metrics = MetricsDictionary()
    norm = {"l0": 0, "l1": 1, "l2": 2, "li": np.inf}[FLAGS.norm]
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy
    nll_fn = tf.keras.metrics.sparse_categorical_crossentropy

    @tf.function
    def get_target(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        r0 = init_r0(image.shape, FLAGS.epsilon, norm, FLAGS.init)
        r0 = project_box(image, r0, 0.0, 1.0)
        logits = test_classifier(image + r0)["logits"]
        target = tf.argmax(
            tf.where(label_onehot == 0, logits,
                     -np.inf * tf.ones_like(logits)),
            axis=-1,
        )
        nll = tf.reduce_mean(nll_fn(label, logits))
        acc = tf.reduce_mean(acc_fn(label, logits))
        return acc, nll, target

    def test_step(image, label):
        image = tf.convert_to_tensor(image)
        label = tf.convert_to_tensor(label)

        # clean accuracy
        logits = test_classifier(image)["logits"]
        test_metrics["acc"](acc_fn(label, logits))
        # random sampling
        targets_prob = tf.zeros((image.shape[0], num_classes))
        mean_acc = 0.0
        mean_nll = 0.0
        for i in range(FLAGS.restarts):
            acc, nll, target = get_target(image, label)
            mean_acc += acc
            mean_nll += nll
            targets_prob += tf.one_hot(target, num_classes)
        targets_prob /= FLAGS.restarts
        d = tfp.distributions.Categorical(probs=targets_prob)
        test_metrics["acc_hat"](acc)
        test_metrics["nll_hat"](nll)
        test_metrics["entropy_hat"](d.entropy())

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(test_ds, 1):
            test_step(image, label)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                break
    except:
        logging.info("Stopping after {}".format(batch_index))
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                 batch_index),
        )


if __name__ == "__main__":
    absl.app.run(main)
