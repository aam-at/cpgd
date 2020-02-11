from __future__ import absolute_import, division, print_function

import importlib
import itertools
import subprocess
from pathlib import Path

import numpy as np
import six

from lib.generate_script import generate_test_optimizer


def generate_test_optimizer_l2(load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_l2_madry', load_from,
                                   **kwargs)


def generate_test_optimizer_l1(load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_l1_madry', load_from,
                                   **kwargs)


if __name__ == '__main__':
    hostname = subprocess.getoutput('hostname')
    pass
