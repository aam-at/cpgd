from __future__ import absolute_import, division, print_function

import argparse
import logging
import sys
import time

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from cleverhans.attacks import CarliniWagnerL2, ElasticNetMethod
from cleverhans.model import Model
from lib.tf_utils import l0_metric, l1_metric, l2_metric, li_metric
from lib.utils import (batch_iterator, format_float,
                       import_func_annotations_as_flags, log_metrics,
                       register_experiment_flags, setup_experiment)

from config import test_thresholds
from data import load_mnist
from models import MadryCNNTf
from utils import load_madry

tf.compat.v1.disable_v2_behavior()

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_cleverhans")
flags.DEFINE_string("attack", None, "attack class")
flags.DEFINE_string("norm", "l2", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
FLAGS = flags.FLAGS

lp_attacks = {
    "l2": {
        'cw': CarliniWagnerL2,
    },
    "l1": {
        'ead': ElasticNetMethod
    }
}


def import_flags(norm, attack):
    global lp_attacks
    assert norm in lp_attacks
    assert attack in lp_attacks[norm]
    import_func_annotations_as_flags(lp_attacks[norm][attack].parse_params,
                                     prefix="attack_",
                                     include_kwargs_with_defaults=True)


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_cleverhans_{FLAGS.attack}_{FLAGS.norm}_test",
                     [__file__])

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    test_images, test_labels = test_ds
    if FLAGS.num_batches != -1:
        total_examples = FLAGS.num_batches * FLAGS.batch_size
        test_images = test_images[:total_examples]
        test_labels = test_labels[:total_examples]

    # models
    num_classes = 10
    classifier = MadryCNNTf()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    lp_metrics = {
        "l2": l2_metric,
        "l1": l1_metric,
    }

    class MadryModel(Model):
        def get_logits(self, x, **kwargs):
            return classifier(x, **kwargs)["logits"]

        def get_probs(self, x, **kwargs):
            return test_classifier(x, **kwargs)["prob"]

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    def test_step(image, image_adv, label):
        outs = test_classifier(image)
        is_corr = tf.equal(outs['pred'], label)
        outs_adv = test_classifier(image_adv)
        is_adv = tf.not_equal(outs_adv["pred"], label)

        # metrics
        nll_loss = nll_loss_fn(label, outs["logits"])
        acc = acc_fn(label, outs["logits"])
        acc_adv = acc_fn(label, outs_adv["logits"])

        # accumulate metrics
        results = {}
        results['nll_loss'] = nll_loss
        results["acc"] = acc
        results["conf"] = outs["conf"]
        results[f"acc_{FLAGS.norm}"] = acc_adv
        results[f"conf_{FLAGS.norm}"] = outs_adv["conf"]

        # measure norm
        r = image - image_adv
        lp = lp_metrics[FLAGS.norm](r)
        l0 = l0_metric(r)
        l1 = l1_metric(r)
        l2 = l2_metric(r)
        li = li_metric(r)
        results["l0"] = l0
        results["l1"] = l1
        results["l2"] = l2
        results["li"] = li
        # exclude incorrectly classified
        results["l0_corr"] = l0[tf.logical_and(is_corr, is_adv)]
        results["l1_corr"] = l1[tf.logical_and(is_corr, is_adv)]
        results["l2_corr"] = l2[tf.logical_and(is_corr, is_adv)]
        results["li_corr"] = li[tf.logical_and(is_corr, is_adv)]

        # robust accuracy at threshold
        for threshold in test_thresholds[f"{FLAGS.norm}"]:
            is_adv_at_th = tf.logical_and(lp <= threshold, is_adv)
            results[f"acc_{FLAGS.norm}_%s" % format_float(threshold)] = ~is_adv_at_th
        results["success_rate"] = is_adv[is_corr]

        return results

    # reset metrics
    start_time = time.time()
    try:
        with tf.compat.v1.Session() as sess:
            # attacks
            attack_kwargs = {
                kwarg.replace("attack_", ""): getattr(FLAGS, kwarg)
                for kwarg in dir(FLAGS) if kwarg.startswith("attack_")
            }
            attack = lp_attacks[FLAGS.norm][FLAGS.attack](MadryModel(), sess)
            attack.parse_params(**attack_kwargs)
            # load classifier
            X_placeholder = tf.keras.layers.Input([28, 28, 1])
            X_adv_placeholder = tf.keras.layers.Input([28, 28, 1])
            y_placeholder = tf.keras.layers.Input([], dtype=tf.int64)
            classifier.build(X_placeholder.shape)
            sess.run(tf.compat.v1.global_variables_initializer())
            load_madry(FLAGS.load_from,
                       classifier.trainable_variables,
                       sess=sess)

            image_adv_list = []
            for image, label in batch_iterator(test_images,
                                               test_labels,
                                               batchsize=FLAGS.batch_size):
                label_onehot = tf.keras.utils.to_categorical(
                    label, num_classes)
                image_adv = attack.generate_np(image, y=label_onehot, **attack_kwargs)
                image_adv_list.append(image_adv)
            image_adv = np.vstack(image_adv_list)

            # compute final results
            results_op = test_step(X_placeholder, X_adv_placeholder,
                                   y_placeholder)
            final_results = {}
            total_examples = {}
            for image, image_adv, label in batch_iterator(
                    test_images,
                    image_adv,
                    test_labels,
                    batchsize=FLAGS.batch_size):
                results = sess.run(
                    results_op, {
                        X_placeholder: image,
                        X_adv_placeholder: image_adv,
                        y_placeholder: label
                    })
                for key, value in results.items():
                    if key not in final_results:
                        final_results[key] = 0
                        assert key not in total_examples
                        total_examples[key] = 0
                    final_results[key] += value.sum()
                    total_examples[key] += len(value)

            class WrapValue:
                def __init__(self, value):
                    self.value = value

                def __repr__(self):
                    return str(self.value)

                def result(self):
                    return self.value

            for key, value in final_results.items():
                final_results[key] /= total_examples[key]
                final_results[key] = WrapValue(final_results[key])
    except KeyboardInterrupt:
        logging.info("Stopping...")
    except Exception as e:
        raise
    finally:
        sess.close()
        e = sys.exc_info()[1]
        if e is None or isinstance(e, KeyboardInterrupt):
            log_metrics(
                final_results,
                "Test results [{:.2f}s]:".format(time.time() - start_time),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default=None, type=str)
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    import_flags(args.norm, args.attack)
    absl.app.run(main)
