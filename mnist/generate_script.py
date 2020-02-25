from __future__ import absolute_import, division, print_function

import itertools
import subprocess
from pathlib import Path

import numpy as np

from lib.generate_script import generate_test_optimizer

models = [
    './models/mnist_weights_plain.mat', './models/mnist_weights_linf.mat',
    './models/mnist_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')


def generate_test_optimizer_l2(load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_l2_madry', load_from,
                                   **kwargs)


def generate_test_optimizer_l1(load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_l1_madry', load_from,
                                   **kwargs)


if __name__ == '__main__':
    hostname = subprocess.getoutput('hostname')
    pass
