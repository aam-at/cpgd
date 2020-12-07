from __future__ import absolute_import, division, print_function

import ast
import importlib
from pathlib import Path

import six


def format_lr_config(lr_config):
    if isinstance(lr_config, str):
        lr_config = ast.literal_eval(lr_config)
    if lr_config["schedule"] == "constant":
        s = f"{lr_config['config']['learning_rate']}"
    elif lr_config["schedule"] == "linear":
        s = (
            f"linear_init{lr_config['config']['initial_learning_rate']}_"
            f"min{lr_config['config']['minimal_learning_rate']}"
        )
    elif lr_config["schedule"] == "exp":
        s = (
            f"exp_init{lr_config['config']['initial_learning_rate']}_"
            f"min{lr_config['config']['minimal_learning_rate']}"
        )
    return s


def format_name(base_name, attack_args):
    attack = attack_args['attack']
    if attack == 'l1g' and attack_args['attack_hard_threshold']:
        attack = f"{attack}_threshold"
    if attack == 'l0':
        attack = f"{attack}_{attack_args['attack_operator']}"
    name = f"""{base_name}_{attack}_{attack_args["attack_loss"]}
{'_multi' if attack_args['attack_loop_multitargeted'] else ''}
_n{attack_args["attack_iterations"]}
_N{attack_args["attack_loop_number_restarts"]}
"""
    name = f"{name}_{'sim' if attack_args['attack_simultaneous_updates'] else 'alt'}"
    name = f"{name}_{attack_args['attack_primal_opt']}"
    if (
        "attack_gradient_preprocessing" in attack_args
        and attack_args["attack_gradient_preprocessing"]
    ):
        name = f"{name}_gprep"
    if "attack_accelerated" in attack_args and attack_args["attack_accelerated"]:
        name = (
            f"{name}_apg_m{attack_args['attack_momentum']}"
            f"{'_adaptive' if attack_args['attack_adaptive_momentum'] else ''}"
        )
    finetune = attack_args["attack_loop_finetune"]
    name = f"{name}_finetune" if finetune else f"{name}_nofinetune"
    # learning rate
    lr_config = attack_args["attack_loop_lr_config"]
    name = f"{name}_lr_{format_lr_config(lr_config)}"
    # finetune learning rate
    if finetune:
        lr_config = attack_args["attack_loop_finetune_lr_config"]
        name = f"{name}_flr_{format_lr_config(lr_config)}"
    name = f"{name}_{attack_args['attack_dual_opt']}"
    # dual learning rate
    dlr_config = attack_args["attack_loop_dual_lr_config"]
    name = f"{name}_dlr_{format_lr_config(dlr_config)}"
    if finetune:
        dlr_config = attack_args["attack_loop_finetune_dual_lr_config"]
        name = f"{name}_fdlr_{format_lr_config(dlr_config)}"
    if not attack_args['attack_dual_ema']:
        name = f"{name}_noema"
    name = f"""{name}_{attack_args['attack_loop_r0_sampling_algorithm']}
_R{attack_args['attack_loop_r0_sampling_epsilon']}
{'_ods' if attack_args['attack_loop_r0_ods_init'] else ''}
_C{attack_args['attack_loop_c0_initial_const']}"""
    return name.replace("\n", "").replace("/", "")


def get_tmpl_str(script_name, **flags):
    """
    script_name: python module where flags are defined
    """
    script_module = importlib.import_module(script_name)
    defined_flags = script_module.FLAGS._flags().keys()
    str_bfr = six.StringIO()
    str_bfr.write("python %(script_name)s.py " % locals())
    for arg_name, arg_value in flags.items():
        assert arg_name in defined_flags, arg_name
        if isinstance(arg_value, (dict, str)):
            str_bfr.write('--{}="{}" '.format(arg_name, arg_value))
        else:
            str_bfr.write("--{}={} ".format(arg_name, arg_value))
    tmpl_str = str_bfr.getvalue()[:-1]
    return tmpl_str


def generate_test_optimizer(script_name, **kwargs):
    tmpl_str = get_tmpl_str(script_name, **kwargs)
    assert Path(f"{script_name}.py").exists()
    return tmpl_str
