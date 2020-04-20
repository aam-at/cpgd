from __future__ import absolute_import, division, print_function

import argparse
import logging
import time
from pathlib import Path

import absl
import numpy as np
import tensorflow as tf
from absl import flags
from tensorboard.plugins.hparams import api as hp

from data import load_mnist
from foolbox.attacks import (L1ProjectedGradientDescentAttack,
                             L2ProjectedGradientDescentAttack,
                             LinfProjectedGradientDescentAttack)
from foolbox.models import TensorFlowModel
from lib.utils import (MetricsDictionary, get_acc_for_lp_threshold,
                       import_kwargs_as_flags, l1_metric, l2_metric, li_metric,
                       log_metrics, make_input_pipeline,
                       register_experiment_flags, reset_metrics, save_images,
                       setup_experiment)
from models import MadryCNN
from utils import load_madry

# general experiment parameters
register_experiment_flags(working_dir="../results/mnist/test_brendel_lp")
flags.DEFINE_string("norm", "l1", "lp-norm attack")
flags.DEFINE_string("load_from", None, "path to load checkpoint from")
flags.DEFINE_integer("attack_random_restarts", 1, "number of random restarts")
flags.DEFINE_float("attack_abs_stepsize", None, "step size for the attack")
# test parameters
flags.DEFINE_integer("num_batches", -1, "number of batches to corrupt")
flags.DEFINE_integer("batch_size", 100, "batch size")
flags.DEFINE_integer("validation_size", 10000, "training size")

# attack parameters

flags.DEFINE_integer("print_frequency", 1, "summarize frequency")

FLAGS = flags.FLAGS

lp_attacks = {
    'l1': L1ProjectedGradientDescentAttack,
    'l2': L2ProjectedGradientDescentAttack,
    'li': LinfProjectedGradientDescentAttack
}


def main(unused_args):
    assert len(unused_args) == 1, unused_args
    assert FLAGS.load_from is not None
    setup_experiment(f"madry_bethge_{FLAGS.norm}_test")

    # data
    _, _, test_ds = load_mnist(FLAGS.validation_size,
                               data_format="NHWC",
                               seed=FLAGS.data_seed)
    test_ds = tf.data.Dataset.from_tensor_slices(test_ds)
    test_ds = make_input_pipeline(test_ds,
                                  shuffle=False,
                                  batch_size=FLAGS.batch_size)

    # models
    classifier = MadryCNN()
    fclassifier = TensorFlowModel(lambda x: classifier(x)['logits'],
                                  bounds=(0.0, 1.0))

    def test_classifier(x, **kwargs):
        return classifier(x, training=False, **kwargs)

    # load classifier
    classifier(np.zeros([1, 28, 28, 1], dtype=np.float32))
    load_madry(FLAGS.load_from, classifier.trainable_variables)
    model_type = Path(FLAGS.load_from).stem.split("_")[-1]

    test_thresholds = {
        'l1': {
            'plain': np.linspace(2, 10, 5),
            'linf': np.linspace(2.5, 12.5, 5),
            'l2': np.linspace(5, 20, 5)
        },
        'l2': {
            'plain': np.linspace(0.5, 2.5, 5),
            'linf': np.linspace(1, 3, 5),
            'l2': np.linspace(1, 3, 5)
        },
        'li': {
            'plain': np.linspace(0.03, 0.11, 5),
            'linf': [0.2, 0.25, 0.3, 0.325, 0.35],
            'l2': np.linspace(0.05, 0.25, 5)
        }
    }
    attack_epsilons = test_thresholds[FLAGS.norm][model_type]

    # attacks
    attack_kwargs = {
        kwarg.replace('attack_', ''): getattr(FLAGS, kwarg)
        for kwarg in dir(FLAGS) if kwarg.startswith('attack_')
        and kwarg not in ['attack_random_restarts']
    }
    olp = lp_attacks[FLAGS.norm](**attack_kwargs)

    test_metrics = MetricsDictionary()

    def test_step(image, label):
        success = tf.ones((len(attack_epsilons), image.shape[0]),
                          dtype=tf.bool)
        for _ in range(FLAGS.attack_random_restarts):
            _, _, scs = olp(fclassifier,
                            image,
                            label,
                            epsilons=attack_epsilons)
            success = tf.logical_and(success, ~scs)

        for i, epsilon in enumerate(attack_epsilons):
            test_metrics[f"acc_{FLAGS.norm}_%.2f" % epsilon](success[i])

    # reset metrics
    reset_metrics(test_metrics)
    X_lp_list = []
    y_list = []
    start_time = time.time()
    try:
        is_completed = False
        for batch_index, (image, label) in enumerate(test_ds, 1):
            test_step(image, label)
            if batch_index % FLAGS.print_frequency == 0:
                log_metrics(
                    test_metrics, "Batch results [{}, {:.2f}s]:".format(
                        batch_index,
                        time.time() - start_time))
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
                    kwarg for kwarg in dir(FLAGS)
                    if kwarg.startswith('attack_')
                ]
                hp_metric_names = [
                    f"final_{FLAGS.norm}", f"final_{FLAGS.norm}_corr"
                ]
                hp_params = [
                    hp.HParam(hp_param_name)
                    for hp_param_name in hp_param_names
                ]
                hp_metrics = [
                    hp.Metric(hp_metric_name)
                    for hp_metric_name in hp_metric_names
                ]
                hp.hparams_config(hparams=hp_params, metrics=hp_metrics)
                hp.hparams({
                    hp_param_name: getattr(FLAGS, hp_param_name)
                    for hp_param_name in hp_param_names
                })
                final_lp = test_metrics[f"{FLAGS.norm}"].result()
                tf.summary.scalar(f"final_{FLAGS.norm}", final_lp, step=1)
                final_lp_corr = test_metrics[f"{FLAGS.norm}_corr"].result()
                tf.summary.scalar(f"final_{FLAGS.norm}_corr",
                                  final_lp_corr,
                                  step=1)
                tf.summary.flush()
    except:
        logging.info("Stopping after {}".format(batch_index))
    finally:
        log_metrics(
            test_metrics,
            "Test results [{:.2f}s, {}]:".format(time.time() - start_time,
                                                 batch_index))
        X_lp_all = tf.concat(X_lp_list, axis=0).numpy()
        y_all = tf.concat(y_list, axis=0).numpy()
        np.savez(Path(FLAGS.working_dir) / 'X_adv', X_adv=X_lp_all, y=y_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=None, type=str)
    args, _ = parser.parse_known_args()
    assert args.norm in ['l1', 'l2', 'li']
    import_kwargs_as_flags(lp_attacks[args.norm].__init__, 'attack_')
    absl.app.run(main)
