from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import absl
import lib
import numpy as np
import tensorflow as tf
from absl import flags
from lib.daa import LinfBLOBAttack, LinfDGFAttack
from lib.tf_utils import (MetricsDictionary, li_metric, make_input_pipeline)
from lib.utils import (batch_iterator, format_float,
                       import_klass_annotations_as_flags, log_metrics,
                       register_experiment_flags, reset_metrics, setup_experiment)

from config import test_thresholds
from data import fbresnet_augmentor, get_imagenet_dataflow
from models import TsiprasCNN
from utils import load_tsipras

# general experiment parameters
register_experiment_flags(working_dir="../results/imagenet/test_daa")
flags.DEFINE_string("method", "blob", "daa method")
flags.DEFINE_string("data_dir", "$IMAGENET_DIR", "path to imagenet dataset")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test paramrs
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_integer("attack_nb_iter", 200, "number of attack iterations")
flags.DEFINE_integer("attack_nb_restarts", 1, "number of attack restarts")

FLAGS = flags.FLAGS

daa_attacks = {'dgf': LinfDGFAttack, 'blob': LinfBLOBAttack}


def import_flags(method):
    global daa_attacks
    assert method in daa_attacks
    import_klass_annotations_as_flags(daa_attacks[method], prefix="attack_")


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    assert FLAGS.data_dir is not None
    if FLAGS.data_dir.startswith("$"):
        FLAGS.data_dir = os.environ[FLAGS.data_dir[1:]]
    setup_experiment(f"madry_daa_{FLAGS.method}_test",
                     [__file__, lib.daa.__file__])

    # data
    augmentors = fbresnet_augmentor(224, training=False)
    val_ds = get_imagenet_dataflow(FLAGS.data_dir,
                                   FLAGS.batch_size,
                                   augmentors,
                                   mode='val')
    val_ds.reset_state()

    # get first N batches
    assert FLAGS.num_batches != -1
    test_images = []
    test_labels = []
    for batch_index, (image, label) in enumerate(val_ds, 1):
        test_images.append(image.numpy())
        test_labels.append(label.numpy())
        if FLAGS.num_batches != -1 and batch_index >= FLAGS.num_batches:
            break
    test_images = np.vstack(test_images)
    test_labels = np.concatenate(test_labels, axis=0)
    test_indx = np.arange(test_images.shape[0])
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels, test_indx))
    # shuffle for DAA attack
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=True,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]
    classifier = TsiprasCNN()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = [FLAGS.batch_size, 224, 224, 3]
    y_shape = [FLAGS.batch_size, num_classes]
    classifier(tf.zeros(X_shape))
    load_tsipras(FLAGS.load_from, classifier.variables)

    # attack arguments
    attack_kwargs = {
        kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
        and kwarg not in ['attack_nb_iter', 'attack_nb_restarts']
    }
    daa = daa_attacks[FLAGS.method](
        lambda x: test_classifier(tf.reshape(x, [-1] + X_shape[1:]))['logits'],
        **attack_kwargs)

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    image_adv_final = tf.Variable(test_images.copy())
    all_indices = tf.range(image_adv_final.shape[0])

    @tf.function
    def attack_step(image, image_adv, label):
        image_adv = daa.perturb(image, image_adv, label)
        assert_op = tf.Assert(
            tf.logical_and(
                tf.reduce_min(image_adv) >= 0,
                tf.reduce_max(image_adv) <= 1.0), [image_adv])
        with tf.control_dependencies([assert_op]):
            return image_adv

    @tf.function
    def test_step(image, image_adv, label):
        outs = test_classifier(image)
        is_corr = outs["pred"] == label
        outs_adv = test_classifier(image_adv)
        is_adv = outs_adv["pred"] != label
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
        li = li_metric(image - image_adv)
        test_metrics["li"](li)
        # exclude incorrectly classified
        test_metrics["li_corr"](li[tf.logical_and(is_corr, is_adv)])

        # robust accuracy at threshold
        for threshold in test_thresholds["li"]:
            is_adv_at_th = tf.logical_and(li <= threshold + 5e-6, is_adv)
            test_metrics["acc_li_%s" % format_float(threshold, 4)](~is_adv_at_th)
        test_metrics["success_rate"](is_adv[is_corr])

    # reset metrics
    start_time = time.time()
    try:
        is_corr0 = []
        for image, label in batch_iterator(test_images,
                                           test_labels,
                                           batchsize=FLAGS.batch_size):
            is_corr0.append(test_classifier(image)['pred'] == label)
        is_corr0 = tf.concat(is_corr0, axis=0)
        for restart_number in range(FLAGS.attack_nb_restarts):
            x_adv = tf.convert_to_tensor(
                test_images +
                tf.random.uniform(test_images.shape, -daa.eps, daa.eps))
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

            for reshuffle in range(int(FLAGS.attack_nb_iter / 10)):
                for (image, label, indx) in test_ds:
                    image_adv = tf.gather(x_adv, tf.expand_dims(indx, 1))
                    image_adv = attack_step(image, image_adv, label)
                    x_adv = tf.tensor_scatter_nd_update(
                        x_adv, tf.expand_dims(indx, 1), image_adv)

            is_adv = []
            for image_adv, label, is_corr in batch_iterator(
                    x_adv.numpy(),
                    test_labels,
                    is_corr0.numpy(),
                    batchsize=FLAGS.batch_size):
                is_adv.append(
                    tf.logical_and(
                        test_classifier(image_adv)['pred'] != label, is_corr))
            is_adv = tf.concat(is_adv, axis=0)
            image_adv_final.scatter_update(
                tf.IndexedSlices(x_adv[is_adv], all_indices[is_adv]))

            # compute accuracy (combining restarts)
            reset_metrics(test_metrics)
            test_ds2 = tf.data.Dataset.from_tensor_slices(
                (test_images, image_adv_final, test_labels))
            test_ds2 = make_input_pipeline(test_ds2,
                                           shuffle=False,
                                           batch_size=FLAGS.batch_size)
            for image, image_adv, label in test_ds2:
                test_step(image, image_adv, label)
            log_metrics(
                test_metrics,
                "Test results after {} restarts [{:.2f}s]:".format(
                    restart_number,
                    time.time() - start_time),
            )
    except KeyboardInterrupt:
        logging.info("Stopping after {} restarts".format(restart_number))
    except Exception as e:
        raise
    finally:
        e = sys.exc_info()[1]
        if e is None or isinstance(e, KeyboardInterrupt):
            log_metrics(
                test_metrics,
                "Test results [{:.2f}s]:".format(time.time() - start_time),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.method)
    absl.app.run(main)
