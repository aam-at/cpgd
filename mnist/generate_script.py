from __future__ import absolute_import, division, print_function

import importlib
import os.path

import numpy as np
import six


def get_tmpl_str(script_name, **flags):
    """
    script_name: python module where flags are defined
    """
    script_module = importlib.import_module(script_name)
    defined_flags = script_module.FLAGS._flags().keys()
    str_bfr = six.StringIO()
    str_bfr.write("python %(script_name)s.py" % locals())
    for arg_name, arg_value in flags.items():
        assert arg_name in defined_flags, arg_name
        str_bfr.write(" --%(arg_name)s=%(arg_value)s" % locals())
    tmpl_str = str_bfr.getvalue()[:-1]
    return tmpl_str


def generate_test_optimizer(script_name, load_from, **kwargs):
    tmpl_str = get_tmpl_str(script_name, **kwargs)
    assert os.path.exists(load_from)
    return "{} --load_from={}".format(tmpl_str, load_from)


def generate_test_optimizer_l2(load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_l2_madry', load_from, **kwargs)


def generate_test_optimizer_l1(load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_l1_madry', load_from, **kwargs)


if __name__ == '__main__':
    pass
