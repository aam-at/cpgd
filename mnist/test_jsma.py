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

from config import test_thresholds
from data import load_mnist
from lib.utils import (MetricsDictionary, l0_metric, log_metrics,
                       make_input_pipeline, random_targets,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_jsma")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters
flags.DEFINE_string("attack_impl", "cleverhans", "JSMA implementation (cleverhans or art)")
flags.DEFINE_float("attack_theta", 1.0, "theta for jsma")
flags.DEFINE_float("attack_gamma", 1.0, "gamma for jsma")
flags.DEFINE_string("attack_targets", "second", "how to select attack target? (choice: 'random', 'second', 'all')")

FLAGS = flags.FLAGS


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_jsma_test", [__file__])

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    num_classes = 10
    classifier = MadryCNN()

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    X_shape = tf.TensorShape([FLAGS.batch_size, 28, 28, 1])
    y_shape = tf.TensorShape([FLAGS.batch_size, num_classes])
    classifier(tf.zeros(X_shape))
    load_madry(FLAGS.load_from, classifier.trainable_variables)

    # saliency map method attack
    if FLAGS.attack_impl == 'cleverhans':
        from cleverhans.attacks.saliency_map_method import SaliencyMapMethod
        from cleverhans.model import Model

        class MadryModel(Model):
            def get_logits(self, x, **kwargs):
                return test_classifier(x, **kwargs)["logits"]

            def get_probs(self, x, **kwargs):
                return test_classifier(x, **kwargs)["prob"]

        jsma = SaliencyMapMethod(MadryModel())
        tf_jsma_generate = tf.function(jsma.generate)

        def update_params(theta_mul=1.0):
            jsma.parse_params(theta=theta_mul * FLAGS.attack_theta,
                              gamma=FLAGS.attack_gamma,
                              clip_min=0.0,
                              clip_max=1.0)

        def jsma_generate(x, y_target):
            y_target = tf.one_hot(y_target, num_classes)
            return tf_jsma_generate(x, y_target=y_target)

    elif FLAGS.attack_impl == 'art':
        from art.attacks import SaliencyMapMethod
        from art.classifiers import TensorFlowV2Classifier

        def art_classifier(x):
            return test_classifier(x)['logits']

        class PatchedTensorflowClassifier(TensorFlowV2Classifier):

            def class_gradient(self, x, label=None, **kwargs):
                import tensorflow as tf

                if label is None or isinstance(label, (int, np.integer)):
                    gradients = super().class_gradient(x,
                                                       label=label,
                                                       **kwargs)
                else:
                    # Apply preprocessing
                    x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)
                    x_preprocessed_tf = tf.convert_to_tensor(x_preprocessed)

                    def grad_targets(x, y_t):
                        batch_indices = tf.range(x.shape[0], dtype=y_t.dtype)
                        y_t_idx = tf.stack([batch_indices, y_t], axis=1)
                        with tf.GradientTape() as tape:
                            tape.watch(x)
                            predictions = self._model(x)
                            prediction = tf.gather_nd(predictions, y_t_idx)

                        return tape.gradient(prediction, x)

                    gradients = grad_targets(x_preprocessed_tf, label)
                    gradients = tf.expand_dims(gradients, axis=1).numpy()

                return gradients


        art_model = PatchedTensorflowClassifier(model=art_classifier,
                                                input_shape=X_shape[1:],
                                                nb_classes=num_classes,
                                                channel_index=3,
                                                clip_values=(0, 1))

        jsma = SaliencyMapMethod(art_model,
                                 theta=FLAGS.attack_theta,
                                 gamma=FLAGS.attack_gamma,
                                 batch_size=FLAGS.batch_size)

        def update_params(theta_mul=1.0):
            jsma.set_params(theta=theta_mul * FLAGS.attack_theta,
                            gamma=FLAGS.attack_gamma)

        def jsma_generate(x, y_target):
            y_target = tf.one_hot(y_target, num_classes)
            x_adv = jsma.generate(x, y_target)
            return x_adv
    else:
        raise ValueError

    nll_loss_fn = tf.keras.metrics.sparse_categorical_crossentropy
    acc_fn = tf.keras.metrics.sparse_categorical_accuracy

    test_metrics = MetricsDictionary()

    def test_jsma_generate(image, label_onehot):
        outs = test_classifier(image)
        is_corr = outs['pred'] == label
        if FLAGS.attack_targets == 'random':
            target = random_targets(num_classes, label_onehot=label_onehot)
            image_adv = jsma_generate(image, target)
        elif FLAGS.attack_targets == 'all':
            indices = tf.argsort(label_onehot)[:, :-1]
            bestlp = tf.where(is_corr, np.inf, 0.0)
            image_adv = tf.identity(image)
            for i in tf.range(num_classes - 1):
                target = indices[:, i]
                image_adv_i = jsma_generate(image, target)
                l0 = l0_metric(image_adv_i - image)
                image_adv = tf.where(tf.reshape(l0 < bestlp, (-1, 1, 1, 1)),
                                     image_adv_i, image_adv)
                bestlp = tf.minimum(bestlp, l0)
        elif FLAGS.attack_targets == 'second':
            masked_logits = tf.where(tf.cast(label_onehot, tf.bool), -np.inf,
                                     outs['logits'])
            target = tf.argsort(masked_logits, direction='DESCENDING')[:, 0]
            image_adv = jsma_generate(image, target)
        return image_adv

    def test_step(image, label):
        label_onehot = tf.one_hot(label, num_classes)
        outs = test_classifier(image)
        is_corr = outs['pred'] == label

        bestlp = np.inf * tf.ones(image.shape[0])
        bestlp = tf.where(is_corr, bestlp, 0.0)
        image_adv = tf.identity(image)
        for mul in [1.0, -1.0]:
            update_params(mul)
            image_adv_ = test_jsma_generate(image, label_onehot)
            is_adv_ = test_classifier(image_adv_)['pred'] != label
            l0_ = tf.where(is_adv_, l0_metric(image - image_adv_), np.inf)
            image_adv = tf.where(tf.reshape(l0_ < bestlp, (-1, 1, 1, 1)), image_adv_, image_adv)
            bestlp = tf.minimum(l0_, bestlp)

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
        l0 = l0_metric(image - image_adv)
        is_adv = outs_l0["pred"] != label
        for threshold in test_thresholds["l0"]:
            is_adv_at_th = tf.logical_and(l0 <= threshold, is_adv)
            test_metrics["acc_l0_%.2f" % threshold](~is_adv_at_th)
        test_metrics["l0"](l0)
        # exclude incorrectly classified
        is_corr = outs["pred"] == label
        test_metrics["l0_corr"](l0[tf.logical_and(is_corr, is_adv)])

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
