from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags

from cleverhans.attacks.saliency_map_method import SaliencyMapMethod
from cleverhans.model import Model
from config import test_thresholds
from data import fbresnet_augmentor, get_imagenet_dataflow
from lib.utils import (MetricsDictionary, import_kwargs_as_flags, l0_metric,
                       l0_pixel_metric, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_jsma")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
flags.DEFINE_string("data_dir", "$IMAGENET_DIR", "path to imagenet dataset")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_bool("attack_l0_pixel_metric", True, "use l0 pixel metric")
flags.DEFINE_float("attack_theta", 1.0, "theta for jsma")
flags.DEFINE_float("attack_gamma", 1.0, "gamma for jsma")
flags.DEFINE_string("attack_targets", "second", "how to select attack target? (choice: 'random', 'second', 'all')")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args

    assert FLAGS.load_from is not None
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(f"madry_jsma_test", [__file__])

    # data
    augmentors = fbresnet_augmentor(224, training=False)
    val_ds = get_imagenet_dataflow(
        FLAGS.data_dir, FLAGS.batch_size,
        augmentors, mode='val')
    val_ds.reset_state()

    # models
    num_classes = len(TsiprasCNN.LABEL_RANGES)
    classifier = TsiprasCNN()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    class MadryModel(Model):
        def get_logits(self, x, **kwargs):
            return test_classifier(x, **kwargs)["logits"]

        def get_probs(self, x, **kwargs):
            return test_classifier(x, **kwargs)["prob"]

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 224, 224, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_tsipras(FLAGS.load_from, classifier.variables)

    # saliency map method attack
    metric = l0_pixel_metric if FLAGS.attack_l0_pixel_metric else l0_metric
    jsma = SaliencyMapMethod(MadryModel())
    jsma.parse_params(theta=FLAGS.attack_theta,
                      gamma=FLAGS.attack_gamma,
                      clip_min=0.0,
                      clip_max=1.0)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        outs = test_classifier(image)
        is_corr = outs['pred'] == label

        if FLAGS.attack_targets == 'random':
            image_adv = jsma.generate(image, y=label_onehot)
        elif FLAGS.attack_targets == 'all':
            indices = tf.argsort(label_onehot)[:, :-1]
            bestlp = tf.where(is_corr, np.inf, 0.0)
            image_adv = tf.identity(image)
            for i in tf.range(num_classes - 1):
                target_onehot = tf.one_hot(indices[:, i], num_classes)
                image_adv_i = jsma.generate(image, y_target=target_onehot)
                l0 = l0_metric(image_adv_i - image)
                image_adv = tf.where(tf.reshape(l0 < bestlp, (-1, 1, 1, 1)),
                                     image_adv_i, image_adv)
                bestlp = tf.minimum(bestlp, l0)
        elif FLAGS.attack_targets == 'second':
            masked_logits = tf.where(tf.cast(label_onehot, tf.bool), -np.inf,
                                     outs['logits'])
            target = tf.argsort(masked_logits, direction='DESCENDING')[:, 0]
            target_onehot = tf.one_hot(target, num_classes)
            image_adv = jsma.generate(image, y_target=target_onehot)

        outs_l0 = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_l0 = acc_fn(label, outs_l0["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l0"](acc_l0)
        test_metrics["conf_l0"](outs_l0["conf"])

        # measure norm
        l0 = metric(image - image_adv)
        is_adv = outs_l0["pred"] != label
        for threshold in test_thresholds["l0"]:
            is_adv_at_th = tf.logical_and(l0 <= threshold, is_adv)
            test_metrics["acc_l0_%.2f" % threshold](~is_adv_at_th)
        test_metrics["l0"](l0)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics["l0_corr"](l0[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(val_ds, 1):
            X_lp = test_step(image, label)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
            save_path = os.path.join(FLAGS.samples_dir,
                                     "epoch_orig-%d.png" % batch_index)
            save_images(image, save_path, data_format="NHWC")
            save_path = os.path.join(FLAGS.samples_dir,
                                     "epoch_l0-%d.png" % batch_index)
            save_images(X_lp, save_path, data_format="NHWC")
            # save adversarial data
            X_lp_list.append(X_lp)
            y_list.append(label)
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                break
    except KeyboardInterrupt:
        logging.info("Stopping after {}".format(batch_index))
    except Exception as e:
        raise
    finally:
        e = sys.exc_info()[1]
        if e is None or isinstance(e, KeyboardInterrupt):
            log_metrics(
                test_metrics,
                "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                     batch_index),
            )
            X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
            y_all = tf.concat(y_list, axis=0).numpy()
            np.savez(Path(FLAGS.working_dir) / "X_adv",
                     X_adv=X_lp_all,
                     y=y_all)


if __name__ == "__main__":
    absl.app.run(main)
