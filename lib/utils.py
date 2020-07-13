from __future__ import absolute_import, division, print_function

import inspect
import json
import logging
import os
import subprocess
import typing
from argparse import Namespace
from shutil import copyfile

import numpy as np
import six
import tensorflow as tf
from absl import flags
from absl.flags import DuplicateFlagError

FLAGS = flags.FLAGS


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            value = self.__getitem__(key)
        except KeyError as exc:
            return None
        if isinstance(value, dict):
            value = AttributeDict(value)
        return value

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


def flags_to_params(fls):
    return Namespace(**{k: f.value for k, f in fls.__flags.items()})


def import_klass_annotations_as_flags(klass,
                                      prefix='',
                                      exclude_args=None,
                                      include_kwargs_with_defaults=False):
    if exclude_args is None:
        exclude_args = []
    imported = []
    for base_klass in klass.mro():
        imported += import_func_annotations_as_flags(
            base_klass.__init__,
            prefix=prefix,
            exclude_args=exclude_args + imported,
            include_kwargs_with_defaults=include_kwargs_with_defaults)


def import_func_annotations_as_flags(f,
                                     prefix='',
                                     exclude_args=None,
                                     include_kwargs_with_defaults=False):
    if exclude_args is None:
        exclude_args = []
    spec = inspect.getfullargspec(f)
    flag_defines = {
        str: flags.DEFINE_string,
        bool: flags.DEFINE_bool,
        int: flags.DEFINE_integer,
        float: flags.DEFINE_float,
    }
    imported = []
    for index, (kwarg, kwarg_type) in enumerate(spec.annotations.items()):
        if kwarg in exclude_args:
            continue
        try:
            kwarg_default = spec.defaults[index]
        except:
            kwarg_default = spec.kwonlydefaults[kwarg]
        is_known_type = False
        arg_type = kwarg_type
        # generic type
        if hasattr(kwarg_type, "__args__"):
            kwarg_types = kwarg_type.__args__
            for type in kwarg_types:
                if type in flag_defines:
                    is_known_type = True
                    arg_type = type
                    break
        else:
            if kwarg_type in flag_defines:
                is_known_type = True
        if not is_known_type:
            logging.debug(f"Uknown {kwarg} type {kwarg_type}")
        arg_name = f"{prefix}{kwarg}"
        try:
            flag_defines[arg_type](arg_name, kwarg_default, f"{kwarg}")
            imported.append(kwarg)
        except DuplicateFlagError as e:
            logging.debug(e)
    if include_kwargs_with_defaults and spec.defaults is not None:
        total_kwargs_with_defaults = len(spec.defaults)
        for index, kwarg in enumerate(spec.args[-total_kwargs_with_defaults:]):
            if kwarg in imported or kwarg in exclude_args:
                continue
            kwarg_default = spec.defaults[index]
            kwarg_type = type(kwarg_default)
            if kwarg_type not in flag_defines:
                logging.debug(f"Uknown {kwarg} type {kwarg_type}")
            else:
                arg_name = f"{prefix}{kwarg}"
                try:
                    flag_defines[kwarg_type](arg_name, kwarg_default,
                                             f"{kwarg}")
                except DuplicateFlagError as e:
                    logging.debug(e)
    return imported


class NanError(BaseException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "NanError: %s has nan value" % self.message


# data utils
def batch_iterator(*args, batchsize=None, shuffle=False):
    assert batchsize is not None
    total_items = len(args[0])
    for arg in args:
        assert len(arg) == total_items

    if shuffle:
        # Shuffles indicies of training data, so we can draw batches
        # from random indicies instead of shuffling whole data
        indx = np.random.permutation(range(total_items))
    else:
        indx = range(total_items)
    for i in range((total_items + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        batch = []
        for arg in args:
            batch.append(arg[indx[sl]])
        yield batch


def select_balanced_subset(X, y, num_classes=10, samples_per_class=10, seed=1):
    total_samples = num_classes * samples_per_class
    X_subset = np.zeros([total_samples] + list(X.shape[1:]), dtype=X.dtype)
    y_subset = np.zeros((total_samples, ), dtype=y.dtype)
    rng = np.random.RandomState(seed)
    for i in range(num_classes):
        yi_indices = np.where(y == i)[0]
        rng.shuffle(yi_indices)
        X_subset[samples_per_class * i:(i + 1) * samples_per_class,
                 ...] = X[yi_indices[:samples_per_class]]
        y_subset[samples_per_class * i:(i + 1) * samples_per_class] = i
    return X_subset, y_subset


def save_images(images, path, data_format="NCHW", **kwargs):
    import torch
    from torchvision.utils import save_image
    if "nrow" not in kwargs:
        kwargs["nrow"] = int(np.sqrt(images.shape[0]))
    if data_format == "NHWC":
        images = np.transpose(images, (0, 3, 1, 2))
    save_image(torch.from_numpy(images), path, **kwargs)


# experiment setup utils
def register_experiment_flags(working_dir="runs", seed=1):
    # experiment parameters
    flags.DEFINE_string("name", None, "name of the experiment")
    flags.DEFINE_integer("seed", seed, "experiment seed")
    flags.DEFINE_integer("data_seed", 1, "experiment seed")
    flags.DEFINE_string("working_dir", working_dir, "path to working dir")
    flags.DEFINE_string("git_revision", None, "git revision")


def setup_experiment(default_name, snapshot_files=None):
    from logging import FileHandler, Formatter, StreamHandler
    np.random.seed(FLAGS.seed)
    if tf.version.VERSION.startswith("1."):
        tf.compat.v1.set_random_seed(FLAGS.seed)
    else:
        tf.random.set_seed(FLAGS.seed)

    dict_values = {k: v.value for k, v in FLAGS._flags().items()}
    if FLAGS.name is None:
        FLAGS.name = default_name % dict_values
    FLAGS.git_revision = get_sha()
    FLAGS.working_dir = prepare_dir(FLAGS.working_dir, FLAGS.name)

    # configure logging
    logger = logging.getLogger()
    [logger.removeHandler(handler) for handler in logger.handlers]
    file_hndl = FileHandler(os.path.join(FLAGS.working_dir, 'tensorflow.log'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)
    cmd_hndl = StreamHandler()
    cmd_hndl.setLevel(logging.INFO)
    cmd_hndl.setFormatter(Formatter('%(message)s'))
    logger.addHandler(cmd_hndl)
    logger.setLevel(logging.DEBUG)

    # print config
    train_params = json.dumps({k: v.value
                               for k, v in FLAGS._flags().items()},
                              sort_keys=True)
    logging.info(train_params)

    if snapshot_files is not None:
        for snapshot_file in snapshot_files:
            copyfile(
                snapshot_file,
                os.path.join(FLAGS.working_dir,
                             os.path.basename(snapshot_file)))


def prepare_dir(dir_path, subdir_name):
    base = os.path.join(dir_path, subdir_name)
    i = 0
    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except OSError:
            i += 1
    return name


# experiment run utils
def log_metrics(metrics, header=None, level=logging.INFO, throw_on_nan=False):
    str_bfr = six.StringIO()
    if header is not None:
        str_bfr.write(header)
    for metric_name, metric_value in metrics.items():
        metric_value = metric_value.result()
        if np.isnan(metric_value):
            if throw_on_nan:
                raise NanError(metric_name)
            else:
                metric_value = -1
        str_bfr.write(" {}: {:.6f},".format(metric_name, metric_value))
    logging.log(level, str_bfr.getvalue()[:-1])


def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset_states()


# git utils
def get_sha(repo='.'):
    """
    Grabs the current SHA-1 hash of the given directory's git HEAD-revision.
    The output of this is equivalent to calling git rev-parse HEAD.

    Be aware that a missing git repository will make this return an error message,
    which is not a valid hash.
    """
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.decode('ascii').strip()
