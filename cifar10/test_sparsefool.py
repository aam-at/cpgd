from __future__ import absolute_import, division, print_function

import logging
import sys
import time
from pathlib import Path

import absl
import torch
import torch.nn.functional as F
from absl import flags
from lib.pt_utils import (MetricsDictionary, l0_metric, l0_pixel_metric,
                          l1_metric, l2_metric, li_metric, setup_torch,
                          to_torch)
from lib.sparsefool import sparsefool
from lib.tf_utils import limit_gpu_growth, make_input_pipeline
from lib.utils import (format_float, import_func_annotations_as_flags,
                       log_metrics, register_experiment_flags, reset_metrics,
                       setup_experiment)

from config import test_thresholds
from data import load_cifar10
from models import MadryCNNPt
from utils import load_madry_pt

# general experiment parameters
register_experiment_flags(working_dir="../results/cifar10/test_sparsefool")
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
    setup_torch(FLAGS.seed)
    setup_experiment("madry_sparsefool_test", [__file__])

    # models
    num_classes = 10
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    classifier = MadryCNNPt(model_type=model_type, wrap_outputs=False)

    # load classifier
    load_madry_pt(FLAGS.load_from,
                  classifier.parameters(),
                  model_type=model_type)
    classifier.cuda()
    classifier.eval()

    # data
    _, _, test_ds = load_cifar10(FLAGS.validation_size,
                                 data_format="NCHW",
                                 seed=FLAGS.data_seed)
    # NOTE: load tensorflow after converting model to cuda
    import tensorflow as tf
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # attacks
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
    }

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        outs = classifier(image, wrap_outputs=True)
        is_corr = outs["pred"] == label

        image_adv = image.clone()
        for indx in torch.where(is_corr)[0]:
            image_i = torch.unsqueeze(image[indx], 0)
            image_adv_i = sparsefool(image_i, classifier, 0.0, 1.0,
                                     **attack_kwargs)[0]
            image_adv[indx] = image_adv_i

        # sanity check
        assert torch.all(
            torch.logical_and(image_adv.min() >= 0,
                              image_adv.max() <= 1.0)), "Outside range"
        outs_adv = classifier(image_adv, wrap_outputs=True)
        is_adv = outs_adv["pred"] != label

        # metrics
        nll_loss = F.cross_entropy(outs["logits"], label, reduction="none")
        acc = outs["pred"] == label
        acc_adv = outs_adv["pred"] == label

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics["acc_adv"](acc_adv)
        test_metrics["conf_adv"](outs_adv["conf"])

        # measure norm
        r = image - image_adv
        rc = r.view(r.shape[0], r.shape[1], -1)
        r = r.view(r.shape[0], -1)
        l0 = l0_metric(r)
        l0p = l0_pixel_metric(rc, channel_dim=1)
        l1 = l1_metric(r)
        l2 = l2_metric(r)
        li = li_metric(r)
        test_metrics["l0"](l0)
        test_metrics["l0p"](l0p)
        test_metrics["l1"](l1)
        test_metrics["l2"](l2)
        test_metrics["li"](li)
        # exclude incorrectly classified
        test_metrics["l0_corr"](l0[torch.logical_and(is_corr, is_adv)])
        test_metrics["l0p_corr"](l0p[torch.logical_and(is_corr, is_adv)])
        test_metrics["l1_corr"](l1[torch.logical_and(is_corr, is_adv)])
        test_metrics["l2_corr"](l2[torch.logical_and(is_corr, is_adv)])
        test_metrics["li_corr"](li[torch.logical_and(is_corr, is_adv)])

        # robust accuracy at threshold
        for threshold in test_thresholds["l1"]:
            is_adv_at_th = torch.logical_and(l1 <= threshold, is_adv)
            test_metrics["acc_l1_%s" % format_float(threshold)](~is_adv_at_th)
        test_metrics["success_rate"](is_adv[is_corr])

        return image_adv

    # reset metrics
    reset_metrics(test_metrics)
    start_time = time.time()
    try:
        for batch_index, (image, label) in enumerate(test_ds, 1):
            X_lp = test_step(*to_torch(image, label))
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(batch_index,
                                                      time.time() -
                                                      start_time),
            )
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


if __name__ == "__main__":
    limit_gpu_growth()
    absl.app.run(main)
