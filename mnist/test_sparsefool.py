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

from config import test_thresholds
from data import load_mnist
from lib.sparsefool import sparsefool
from lib.utils import (MetricsDictionary, add_default_end_points,
                       import_func_annotations_as_flags, l1_metric,
                       limit_gpu_growth, log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNNPt
from utils import load_madry_pt

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_sparsefool")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
import_func_annotations_as_flags(sparsefool, "attack_")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_sparsefool_test", [__file__])

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NCHW",
                               seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    def to_torch(*args, cuda=True):
        torch_tensors = [torch.from_numpy(a.numpy()) for a in args]
        return [t.cuda() if cuda else t for t in torch_tensors]

    # models
    num_classes = 10
    classifier = MadryCNNPt()

    # load classifier
    load_madry_pt(FLAGS.load_from, classifier.parameters())
    classifier.cuda()
    classifier.eval()

    def test_classifier(x):
        logits = classifier(*to_torch(x))
        if logits.is_cuda:
            logits = logits.cpu()
        return add_default_end_points({'logits': tf.convert_to_tensor(logits.detach().numpy())})

    # attacks
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = test_classifier(image)

        batch_indices = tf.range(image.shape[0])
        is_corr = outs['pred'] == label
        image_adv = tf.identity(image)
        for indx in batch_indices[is_corr]:
            image_i = tf.expand_dims(image[indx], 0)
            image_pt_i = torch.from_numpy(image_i.numpy()).to("cuda")
            image_adv_pt_i = sparsefool(image_pt_i, classifier, 0.0, 1.0,
                                        **attack_kwargs)[0]
            image_adv = tf.tensor_scatter_nd_update(
                image_adv, tf.expand_dims([indx], axis=1),
                image_adv_pt_i.detach().cpu().numpy())
        # safety check
        assert tf.reduce_all(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0)), "Outside range"

        outs_adv = test_classifier(image_adv)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_adv = acc_fn(label, outs_adv["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_l1"](acc_adv)
        test_metrics["conf_l1"](outs_adv["conf"])

        # measure norm
        lp = l1_metric(image - image_adv)
        is_adv = outs_adv["pred"] != label
        for threshold in test_thresholds["l1"]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            test_metrics[f"acc_l1_%.2f" % threshold](~is_adv_at_th)
        test_metrics[f"l1"](lp)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics[f"l1_corr"](lp[tf.logical_and(is_corr, is_adv)])
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(test_ds, 1):
            X_lp = test_step(image, label)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
            save_path = os.path.join(FLAGS.samples_dir,
                                     "epoch_orig-%d.png" % batch_index)
            save_images(image.numpy(), save_path, data_format="NCHW")
            save_path = os.path.join(
                FLAGS.samples_dir, f"epoch_l1-%d.png" % batch_index)
            save_images(X_lp.numpy(), save_path, data_format="NCHW")
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
