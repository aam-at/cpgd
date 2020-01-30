from __future__ import absolute_import, division, print_function

import glob
import importlib
import inspect
import os

import decorator
import numpy as np
import six
from absl import flags

flags.DEFINE_boolean("train", True, "train models")
flags.DEFINE_boolean("carlini", False, "test models using Carlini l2-attack")

FLAGS = flags.FLAGS


@decorator.decorator
def concat_commands(f, *args, **kwargs):
    commands = f(*args, **kwargs)
    print("\n".join(commands))


def get_tmpl_str(f, script_name, add_args=None, exclude_args=None):
    """
    script_name: python module where flags are defined
    add_args: additional flags which are not included in the function definition.
    exclude_args: flags to exclude from function signature
    """
    if add_args is None:
        add_args = []
    if exclude_args is None:
        exclude_args = []
    argspec = inspect.getfullargspec(f)
    arg_names = argspec.args
    if len(add_args) > 0:
        for arg in add_args:
            if isinstance(arg, (tuple, list)) > 0:
                arg_names.insert(arg[0], arg[1])
            else:
                arg_names.append(arg)

    script_module = importlib.import_module(script_name)
    defined_flags = script_module.FLAGS._flags().keys()
    str_bfr = six.StringIO()
    str_bfr.write("python %(script_name)s.py " % locals())
    for arg_name in arg_names:
        if arg_name not in exclude_args:
            assert arg_name in defined_flags, arg_name
            str_bfr.write("--%(arg_name)s=%%(%(arg_name)s)s " % locals())
    tmpl_str = str_bfr.getvalue()[:-1]
    return tmpl_str


@concat_commands
def generate_test(root_dir,
                  working_dir=None,
                  attack_iter=50,
                  attack_clip=0.5,
                  attack_overshoot=0.02,
                  sort_labels=True,
                  filter_dirs=False):
    tmpl_str = get_tmpl_str(
        generate_test,
        'test',
        add_args=[(0, 'load_from')],
        exclude_args=['root_dir', 'filter_dirs'])
    working_dirs = glob.glob(working_dir + '/*')
    for load_from in sorted(glob.glob(root_dir)):
        if working_dir is None:
            working_dir = os.path.abspath(os.path.join(load_from, os.pardir))
            working_dir = working_dir.replace("runs", "test")
        else:
            working_dir = working_dir % locals()
        working_dirs = [
            os.path.basename(working_path)[:-1]
            for working_path in glob.glob(os.path.join(working_dir, '*'))
        ]
        if not filter_dirs or os.path.basename(load_from) not in working_dirs:
            yield tmpl_str % locals()


@concat_commands
def generate_test_corruptions(root_dir,
                              working_dir=None,
                              filter_dirs=False):
    tmpl_str = get_tmpl_str(
        generate_test_corruptions,
        'test_corruptions',
        add_args=[(0, 'load_from')],
        exclude_args=['root_dir', 'filter_dirs'])
    working_dirs = glob.glob(working_dir + '/*')
    for load_from in sorted(glob.glob(root_dir)):
        if working_dir is None:
            working_dir = os.path.abspath(os.path.join(load_from, os.pardir))
            working_dir = working_dir.replace("runs", "test")
        else:
            working_dir = working_dir % locals()
        working_dirs = [
            os.path.basename(working_path)[:-1]
            for working_path in glob.glob(os.path.join(working_dir, '*'))
        ]
        if not filter_dirs or os.path.basename(load_from) not in working_dirs:
            yield tmpl_str % locals()


@concat_commands
def generate_test_carlini(root_dir, working_dir=None, batch_size=100,
                          carlini_max_iter=10000, carlini_confidence=0,
                          use_carlini_prob=False, carlini_binary_steps=9,
                          carlini_lb=0.0, carlini_ub=1e6,
                          generate_summary=True, sort_labels=True,
                          filter_dirs=False):
    tmpl_str = get_tmpl_str(generate_test_carlini, 'test_carlini_l2', add_args=[(0, 'load_from')],
                            exclude_args=['root_dir', 'filter_dirs'])
    working_dirs = glob.glob(working_dir + '/*')
    for load_from in sorted(glob.glob(root_dir)):
        if not load_from.endswith('_0'):
            continue
        if working_dir is None:
            working_dir = os.path.abspath(os.path.join(load_from, os.pardir))
            working_dir = working_dir.replace("runs", "test") + '_ca'
        working_dirs = [os.path.basename(working_path) for working_path in
                        glob.glob(os.path.join(working_dir, '*'))]
        if not filter_dirs or os.path.basename(load_from) not in working_dirs:
            yield tmpl_str % locals()


@concat_commands
def generate_gan(name='%(gan_loss)s_%(dsc_penalty)s_%(dsc_lmbd_grad)s_',
                 gan_loss='non_saturating', dsc_penalty='wgangp',
                 dsc_lmbd_grad=0.0, niter=100, train_dir='runs_gan',
                 seed=1, runs=1):
    np.random.seed(seed)
    tmpl_str = get_tmpl_str(
        generate_gan, 'train_gan', exclude_args=['runs'])
    name = name % locals()
    for i in range(runs):
        seed = np.random.randint(1234)
        yield tmpl_str % locals()


if __name__ == '__main__':
    pass
