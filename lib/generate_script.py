from __future__ import absolute_import, division, print_function

import importlib
from pathlib import Path

import six


def get_tmpl_str(script_name, ignored_flags, **flags):
    """
    script_name: python module where flags are defined
    """
    script_module = importlib.import_module(script_name)
    defined_flags = script_module.FLAGS._flags().keys()
    str_bfr = six.StringIO()
    str_bfr.write("python %(script_name)s.py " % locals())
    for arg_name, arg_value in flags.items():
        assert arg_name in defined_flags or arg_name in ignored_flags, arg_name
        str_bfr.write("--%(arg_name)s=%(arg_value)s " % locals())
    tmpl_str = str_bfr.getvalue()[:-1]
    return tmpl_str


def generate_test_optimizer(script_name,
                            norm,
                            load_from,
                            ignored_flags={},
                            **kwargs):
    tmpl_str = get_tmpl_str(script_name, ignored_flags, **kwargs)
    assert Path(f"{script_name}.py").exists()
    assert Path(load_from).exists()
    return "{} --norm={} --load_from={}".format(tmpl_str, norm, load_from)
