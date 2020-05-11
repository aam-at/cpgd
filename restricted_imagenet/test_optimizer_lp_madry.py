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
from tensorboard.plugins.hparams import api as hp

import lib
from data import get_imagenet_dataflow, fbresnet_augmentor
from lib.attack_l0 import ProximalL0Attack
from lib.attack_l1 import GradientL1Attack, ProximalL1Attack
from lib.attack_l2 import GradientL2Attack, ProximalL2Attack
from lib.attack_li import ProximalLiAttack
from lib.attack_utils import AttackOptimizationLoop
from lib.utils import (MetricsDictionary, get_acc_for_lp_threshold,
                       import_klass_kwargs_as_flags, log_metrics,
                       register_experiment_flags,
                       reset_metrics, save_images, setup_experiment)
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_lp")
flags.DEFINE_string(
    "attack", None, "choice of the attack ('l0', 'l1', 'l2', 'l2g', 'li')"
)
flags.DEFINE_string("data_dir", "$IMAGENET_DIR", "path to imagenet dataset")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")

# attack parameters
import_klass_kwargs_as_flags(AttackOptimizationLoop, "attack_loop_")

FLAGS = flags.FLAGS

lp_attacks = {
    "l0": ("l0", ProximalL0Attack),
    "l1": ("l1", ProximalL1Attack),
    "l1g": ("l1", GradientL1Attack),
    "l2": ("l2", ProximalL2Attack),
    "l2g": ("l2", GradientL2Attack),
    "li": ("li", ProximalLiAttack),
}


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    norm, attack_klass = lp_attacks[FLAGS.attack]

    assert FLAGS.load_from is not None
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(
        f"madry_{norm}_test",
        [
            lib.attack_lp.__file__,
            getattr(lib, f"attack_{norm}").__file__,
            lib.attack_utils.__file__,
            lib.utils.__file__,
        ],
    )

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

    # attacks
    attack_loop_kwargs = {
        kwarg.replace("attack_loop_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS)
        if kwarg.startswith("attack_loop_")
    }
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS)
        if kwarg.startswith("attack_") and not kwarg.startswith("attack_loop_")
    }
    alp = attack_klass(lambda x: test_classifier(x)["logits"], **attack_kwargs)
    alp.build([X_shape, y_shape])
    allp = AttackOptimizationLoop(alp, **attack_loop_kwargs)

    # test metrics
    test_thresholds = {
        "l0": [2, 4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 45],
        'l1': [5, 15, 16, 25, 27, 38, 40, 49, 50, 60, 100, 150, 200, 250],
        'l2': [0.2, 0.4, 0.6, 0.8, 1.0, 2, 3, 4, 5, 6],
        'li': np.array([0.25, 0.5, 0.75, 1, 1.25, 2, 4, 6, 8, 10]) / 255.0
    }
    test_metrics = MetricsDictionary()

    @tf.function
    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        image_lp = allp.run_loop(image, label_onehot)

        outs = test_classifier(image)
        outs_lp = test_classifier(image_lp)

        # metrics
        nll_loss = tf.keras.metrics.sparse_categorical_crossentropy(
            label, outs["logits"]
        )
        acc_fn = tf.keras.metrics.sparse_categorical_accuracy
        label = tf.convert_to_tensor(label)
        acc = acc_fn(label, outs["logits"])
        acc_lp = acc_fn(label, outs_lp["logits"])

        # accumulate metrics
        test_metrics["nll_loss"](nll_loss)
        test_metrics["acc"](acc)
        test_metrics["conf"](outs["conf"])
        test_metrics[f"acc_{norm}"](acc_lp)
        test_metrics[f"conf_{norm}"](outs_lp["conf"])
        
        # measure norm
        lp = alp.lp_metric(image - image_lp)
        for threshold in test_thresholds[norm]:
            acc_th = get_acc_for_lp_threshold(
                lambda x: test_classifier(x)["logits"],
                image,
                image_lp,
                label,
                lp,
                threshold,
            )
            test_metrics[f"acc_{norm}_%.4f" % threshold](acc_th)
        test_metrics[f"{norm}"](lp)
        # compute statistics only for correctly classified inputs
        is_corr = outs["pred"] == label
        test_metrics[f"{norm}_corr"](lp[is_corr])

        return image_lp

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
    start_time = time.time()
    try:
        is_completed = False
        val_ds.reset_state()
        for batch_index, (image, label) in enumerate(val_ds, 1):
            X_lp = test_step(image, label)
            log_metrics(
                test_metrics,
                "Batch results [{}, {:.2f}s]:".format(
                    batch_index, time.time() - start_time
                ),
            )
            # save adversarial data
            save_path = os.path.join(
                FLAGS.samples_dir, "epoch_orig-%d.png" % batch_index
            )
            save_images(image, save_path, data_format="NHWC")
            save_path = os.path.join(
                FLAGS.samples_dir, f"epoch_{norm}-%d.png" % batch_index
            )
            save_images(X_lp, save_path, data_format="NHWC")
            X_lp_list.append(X_lp)
            y_list.append(label)
            if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
                is_completed = True
                break
        else:
            is_completed = True
        if is_completed:
            # hyperparameter tuning
            with tf.summary.create_file_writer(FLAGS.working_dir).as_default():
                # hyperparameters
                hp_param_names = [
                    kwarg for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
                ]
                hp_metric_names = [f"final_{norm}", f"final_{norm}_corr"]
                hp_params = [
                    hp.HParam(hp_param_name) for hp_param_name in hp_param_names
                ]
                hp_metrics = [
                    hp.Metric(hp_metric_name) for hp_metric_name in hp_metric_names
                ]
                hp.hparams_config(hparams=hp_params, metrics=hp_metrics)
                hp.hparams(
                    {
                        hp_param_name: getattr(FLAGS, hp_param_name)
                        for hp_param_name in hp_param_names
                    }
                )
                final_lp = test_metrics[f"{norm}"].result()
                tf.summary.scalar(f"final_{norm}", final_lp, step=1)
                final_lp_corr = test_metrics[f"{norm}_corr"].result()
                tf.summary.scalar(f"final_{norm}_corr", final_lp_corr, step=1)
                tf.summary.flush()
    except KeyboardInterrupt:
        logging.info("Stopping after {}".format(batch_index))
    except:
        raise
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time, batch_index),
        )
        X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / "X_adv", X_adv=X_lp_all, y=y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default=None, type=str)
    args, _ = parser.parse_known_args()
    assert args.attack in lp_attacks
    attack_klass = lp_attacks[args.attack][1]
    import_klass_kwargs_as_flags(attack_klass, "attack_")
    absl.app.run(main)
