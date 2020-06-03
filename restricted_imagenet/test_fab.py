from __future__ import absolute_import, division, print_function

import logging
import os
import sys
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
import torch
from absl import flags

import lib
from config import test_thresholds
from data import fbresnet_augmentor, get_imagenet_dataflow
from lib.fab import FABAttack, FABModelAdapter
from lib.utils import (MetricsDictionary, import_klass_annotations_as_flags,
                       l0_metric, l1_metric, l2_metric, li_metric,
                       limit_gpu_growth, log_metrics, make_input_pipeline,
                       random_targets, register_experiment_flags,
                       reset_metrics, save_images, setup_experiment)
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/imagenet/test_fab")
flags.DEFINE_string("data_dir", "$IMAGENET_DIR", "path to imagenet dataset")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
import_klass_annotations_as_flags(FABAttack, "attack_")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(f"madry_fab_test", [__file__, lib.fab.__file__])

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

    lp_metrics = {
        "l1": l1_metric,
        "l2": l2_metric,
        "li": li_metric
    }

    # attacks
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }
    fab = FABAttack(
        model=FABModelAdapter(lambda x: test_classifier(x)['logits']),
        seed=FLAGS.seed,
        **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)
        is_corr = outs['pred'] == label

        # tensorflow model + pytorch attack
        image_pt = torch.from_numpy(image.numpy()).cuda()
        label_pt = torch.from_numpy(label.numpy()).cuda()
        image_adv_pt = fab.perturb(image_pt, label_pt)
        image_adv = image_adv_pt.cpu().numpy()
        outs_lp = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_lp = acc_fn(label, outs_lp["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_{FLAGS.attack_norm}"](acc_lp)
        test_metrics[f"conf_{FLAGS.attack_norm}"](outs_lp["conf"])

        # measure norm
        lp = lp_metrics[FLAGS.attack_norm](image - image_adv)
        is_adv = outs_lp["pred"] != label
        for threshold in test_thresholds[f"{FLAGS.attack_norm}"]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            test_metrics[f"acc_{FLAGS.attack_norm}_%.4f" % threshold](~is_adv_at_th)
        test_metrics[f"{FLAGS.attack_norm}"](lp)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics[f"{FLAGS.attack_norm}_corr"](lp[tf.logical_and(is_corr, is_adv)])

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
    limit_gpu_growth()
    absl.app.run(main)
