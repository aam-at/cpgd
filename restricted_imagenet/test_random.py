from __future__ import absolute_import, division, print_function

import logging
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

from data import get_imagenet_dataflow, fbresnet_augmentor
from lib.attack_utils import init_r0, project_box
from lib.utils import (MetricsDictionary, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics,
                       setup_experiment)
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/imagenet/test_random_lp")
flags.DEFINE_string("data_dir", None, "path to imagenet dataset")
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
    augmentors = fbresnet_augmentor(224, training=False)
    val_ds = get_imagenet_dataflow(
        FLAGS.data_dir, FLAGS.batch_size,
        augmentors, mode='val')

    # models
    num_classes = len(TsiprasCNN.LABEL_RANGES)
    classifier = TsiprasCNN()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 224, 224, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_tsipras(FLAGS.load_from, classifier.variables)

    # test metrics
    test_metrics = MetricsDictionary()
    norm = {"l0": 0, "l1": 1, "l2": 2, "li": np.inf}[FLAGS.norm]

    @tf.function
    def get_target(image, label_onehot):
        r0 = init_r0(image.shape, FLAGS.epsilon, norm, FLAGS.init)
        r0 = project_box(image, r0, 0.0, 1.0)
        logits = test_classifier(image + r0)["logits"]
        target_indx = tf.argmax(
            tf.where(label_onehot == 0, logits, -np.inf * tf.ones_like(logits)),
            axis=-1,
        )
        return target_indx

    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        targets_prob = tf.zeros_like(label_onehot)
        for i in range(FLAGS.restarts):
            target_indx = get_target(image, label_onehot)
            targets_prob += tf.one_hot(target_indx, num_classes)
        targets_prob /= FLAGS.restarts
        d = tfp.distributions.Categorical(probs=targets_prob)
        test_metrics["entropy"](d.entropy())

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        val_ds.reset_state()
        for batch_index, (image, label) in enumerate(val_ds, 1):
            test_step(image, label)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(
                    batch_index, time.time() - start_time
                ),
            )
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                break
    except:
        logging.info("Stopping after {}".format(batch_index))
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time, batch_index),
        )


if __name__ == "__main__":
    absl.app.run(main)
