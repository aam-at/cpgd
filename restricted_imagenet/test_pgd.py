from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from cleverhans.attacks import ProjectedGradientDescent, SparseL1Descent
from cleverhans.model import Model

from config import test_thresholds
from data import fbresnet_augmentor, get_imagenet_dataflow
from lib.utils import (MetricsDictionary, import_func_annotations_as_flags,
                       l1_metric, l2_metric, li_metric, log_metrics,
                       make_input_pipeline, register_experiment_flags,
                       reset_metrics, save_images, setup_experiment)
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/imagenet/test_pgd")
flags.DEFINE_string("norm", "lp", "lp-norm attack")
flags.DEFINE_string("data_dir", "$IMAGENET_DIR", "path to imagenet dataset")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test paramrs
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("attack_nb_restarts", "1", "number of attack restarts")

FLAGS = flags.FLAGS

lp_attacks = {
    "l1": SparseL1Descent,
    "l2": ProjectedGradientDescent,
    "li": ProjectedGradientDescent,
}


def import_flags(norm):
    global lp_attacks
    assert norm in lp_attacks
    exclude_args = ['clip_min', 'clip_max', 'rand_init']
    if norm != 'l1':
        exclude_args.append('ord')
    import_func_annotations_as_flags(
        lp_attacks[norm].parse_params,
        prefix="attack_",
        exclude_args=exclude_args,
        include_kwargs_with_defaults=True)


def main(unused_args):
    assert len(unused_args) == 1, unused_args

    assert FLAGS.load_from is not None
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(f"madry_pgd_{FLAGS.norm}_test", [__file__])

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

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 224, 224, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_tsipras(FLAGS.load_from, classifier.variables)

    # lp sparse PGD
    class MadryModel(Model):
        def get_logits(self, x, **kwargs):
            return test_classifier(x, **kwargs)["logits"]

        def get_probs(self, x, **kwargs):
            return test_classifier(x, **kwargs)["prob"]

    pgd = lp_attacks[FLAGS.norm](MadryModel())

    lp_metrics = {
        "l1": l1_metric,
        "l2": l2_metric,
        "li": li_metric
    }

    # attack arguments
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }
    if FLAGS.norm != 'l1':
        attack_kwargs['ord'] = 2 if FLAGS.norm == 'l2' else np.inf
    pgd.parse_params(**attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        outs = test_classifier(image)
        is_corr = test_classifier(image)['pred'] == label

        # run attack on correctly classified points
        batch_indices = tf.range(image.shape[0])
        image_adv = tf.identity(image)
        for _ in tf.range(FLAGS.attack_nb_restarts):
            is_adv = test_classifier(image_adv)['pred'] != label
            image_adv = tf.tensor_scatter_nd_update(
                image_adv, tf.expand_dims(batch_indices[~is_adv], axis=1),
                pgd.generate(image[~is_adv],
                             y=label_onehot[~is_adv],
                             clip_min=0.0,
                             clip_max=1.0,
                             rand_init=True,
                             **attack_kwargs))
        assert_op = tf.Assert(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0), [image_adv])
        with tf.control_dependencies([assert_op]):
            outs_adv = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_adv = acc_fn(label, outs_adv["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_adv"](acc_adv)
        test_metrics["conf_adv"](outs_adv["conf"])

        # measure norm
        # NOTE: cleverhans lp-norm projection may result in numerical error
        # add small constant eps = 1e-6
        lp = lp_metrics[FLAGS.norm](image - image_adv)
        is_adv = outs_adv["pred"] != label
        for threshold in test_thresholds[f"{FLAGS.norm}"]:
            is_adv_at_th = tf.logical_and(lp <= threshold + 5e-6, is_adv)
            test_metrics[f"acc_{FLAGS.norm}_%.4f" % threshold](~is_adv_at_th)
        test_metrics[f"{FLAGS.norm}"](lp)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics[f"{FLAGS.norm}_corr"](lp[tf.logical_and(is_corr, is_adv)])
        test_metrics["success_rate"](is_adv[is_corr])

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
            save_path = os.path.join(
                FLAGS.samples_dir, f"epoch_{FLAGS.norm}-%d.png" % batch_index)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.norm)
    absl.app.run(main)