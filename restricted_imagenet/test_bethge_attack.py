from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from foolbox.attacks import (DatasetAttack, L0BrendelBethgeAttack,
                             L1BrendelBethgeAttack, L2BrendelBethgeAttack,
                             LinearSearchBlendedUniformNoiseAttack,
                             LinfinityBrendelBethgeAttack)
from foolbox.models import TensorFlowModel

from data import fbresnet_augmentor, get_imagenet_dataflow
from lib.utils import (MetricsDictionary, import_kwargs_as_flags, l0_metric,
                       l0_pixel_metric, l1_metric, l2_metric, li_metric,
                       log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/imagenet/test_brendel_lp")
flags.DEFINE_string("norm", "l1", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_string("attack_init", "linear_search", "attack to init Bethge attack")
flags.DEFINE_integer("attack_init_steps", 10000, "number of steps for init attack")
flags.DEFINE_bool("attack_l0_pixel_metric", True, "use l0 pixel metric")

FLAGS = flags.FLAGS

lp_attacks = {
    "l0": L0BrendelBethgeAttack,
    "l1": L1BrendelBethgeAttack,
    "l2": L2BrendelBethgeAttack,
    "li": LinfinityBrendelBethgeAttack,
}


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None

    assert FLAGS.load_from is not None
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(f"madry_bethge_{FLAGS.norm}_test", [__file__])

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

    fclassifier = TensorFlowModel(lambda x: test_classifier(x)["logits"],
                                  bounds=np.array((0.0, 1.0),
                                                  dtype=np.float32))

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 32, 32, 3])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from,
               classifier.trainable_variables,
               model_type=model_type)

    lp_metrics = {
        "l0": l0_pixel_metric if FLAGS.attack_l0_pixel_metric else l0_metric,
        "l1": l1_metric,
        "l2": l2_metric,
        "li": li_metric,
    }
    test_thresholds = {
        "l0": np.linspace(10, 250, 25),
        'l1': [5, 15, 16, 25, 27, 38, 40, 49, 50, 60, 100, 150, 200, 250],
        'l2': [0.2, 0.4, 0.6, 0.8, 1.0, 2, 3, 4, 5, 6],
        'li': np.array([0.25, 0.5, 0.75, 1, 1.25, 2, 4, 6, 8, 10]) / 255.0
    }
    # attacks
    if FLAGS.attack_init == "dataset":
        a0 = DatasetAttack()
        for batch_index, (image, label) in enumerate(test_ds, 1):
            a0.feed(fclassifier, image)
    elif FLAGS.attack_init == "linear_search":
        a0 = LinearSearchBlendedUniformNoiseAttack(
            steps=FLAGS.attack_init_steps)
    else:
        raise ValueError()
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_") if kwarg not in
        ["attack_init", "attack_init_steps", "attack_l0_pixel_metric"]
    }
    olp = lp_attacks[FLAGS.norm](init_attack=a0, **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        image_lp, _, _ = olp(fclassifier, image, label, epsilons=None)

        outs = test_classifier(image)
        outs_lp = test_classifier(image_lp)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_lp = acc_fn(label, outs_lp["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_{FLAGS.norm}"](acc_lp)
        test_metrics[f"conf_{FLAGS.norm}"](outs_lp["conf"])

        # measure norm
        lp = lp_metrics[FLAGS.norm](image - image_lp)
        is_adv = outs_lp["pred"] != label
        for threshold in test_thresholds[FLAGS.norm]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            test_metrics[f"acc_{FLAGS.norm}_%.4f" % threshold](~is_adv_at_th)
        test_metrics[f"{FLAGS.norm}"](lp)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics[f"{FLAGS.norm}_corr"](lp[is_corr])

        return image_lp

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
            save_images(image, save_path, data_format="NHWC")
            save_path = os.path.join(
                FLAGS.samples_dir, f"epoch_{FLAGS.norm}-%d.png" % batch_index)
            save_images(X_lp, save_path, data_format="NHWC")
            # save adversarial data
            X_lp_list.append(X_lp)
            y_list.append(label)
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
        X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / "X_adv", X_adv=X_lp_all, y=y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    assert args.norm in lp_attacks
    import_kwargs_as_flags(lp_attacks[args.norm].__init__, "attack_")
    absl.app.run(main)
