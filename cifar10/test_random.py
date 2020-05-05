from __future__ import absolute_import, division, print_function

import logging
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import flags

from data import load_cifar10
from lib.attack_utils import init_r0, project_box
from lib.utils import (MetricsDictionary, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_random_lp")
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
    _, _, test_ds = load_cifar10(
        FLAGS.validation_size, data_format="NHWC", seed=FLAGS.data_seed
    )
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds, shuffle=False, batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    classifier = MadryCNN(model_type=model_type)

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 32, 32, 3])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    # test metrics
    test_metrics = MetricsDictionary()
    norm = {"l0": 0, "l1": 1, "l2": 2, "li": np.inf}[FLAGS.norm]

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        targets_prob = tf.zeros_like(label_onehot)
        for i in range(FLAGS.restarts):
            r0 = init_r0(image.shape, FLAGS.epsilon, norm, FLAGS.init)
            r0 = project_box(image, r0, 0.0, 1.0)
            logits = test_classifier(image + r0)["logits"]
            target_indx = tf.argmax(
                tf.where(label_onehot == 0, logits, -np.inf * tf.ones_like(logits)),
                axis=-1,
            )
            targets_prob += tf.one_hot(target_indx, num_classes)
        targets_prob /= FLAGS.restarts
        d = tfp.distributions.Categorical(probs=targets_prob)
        test_metrics["entropy"](d.entropy())

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(test_ds, 1):
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
