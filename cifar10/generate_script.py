"""This script can be used to generate test configs for the compared attacks.
"""
from __future__ import absolute_import, division, print_function

import ast
import glob
import importlib
import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from absl import flags
from lib.attack_lp import ProximalPrimalDualGradientAttack
from lib.generate_script import (cleanflags, count_number_of_lines,
                                 format_name, generate_test_optimizer)
from lib.parse_logs import parse_log
from lib.tf_utils import ConstantDecay, ExpDecay, LinearDecay
from lib.utils import (format_float, import_func_annotations_as_flags,
                       import_klass_annotations_as_flags)

from config import test_model_thresholds

models = [
    './models/cifar10_weights_plain.mat', './models/cifar10_weights_linf.mat',
    './models/cifar10_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')
basedir = "results_cifar10"

FLAGS = flags.FLAGS


def test_random(runs=1, master_seed=1):
    existing_names = []
    for model, N, norm, eps, init in itertools.product(
            models, [100], ["l2"], np.linspace(0.05, 0.5, 10),
        ["uniform", "sign"]):
        type = Path(model).stem.split("_")[-1]
        eps = round(eps, 3)
        name = f"cifar10_{type}_N{N}_{init}_{eps}_"
        working_dir = f"../results/cifar10_10/test_random_{type}_{norm}"
        attack_args = {
            'working_dir': working_dir,
            'load_from': model,
            "norm": norm,
            "restarts": N,
            "init": init,
            'epsilon': eps,
            'name': name
        }
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        np.random.seed(master_seed)
        for i in range(runs):
            seed = np.random.randint(1000)
            attack_args["seed"] = seed
            print(generate_test_optimizer('test_random', **attack_args))


def load_logs(load_dir):
    logs = []
    for dir in glob.glob(load_dir + "*"):
        log = parse_log(dir, export_test_params=True)
        logs.append(log)
    df = pd.concat(logs, ignore_index=True)
    return df


@cleanflags
def test_our_attack_config(attack, epsilon=None, seed=123):
    if epsilon is not None:
        import test_our_eps_attack
        from test_our_eps_attack import import_flags, lp_attacks

        flags.FLAGS._flags().clear()
        importlib.reload(test_our_eps_attack)
        script_name = "test_our_eps_attack"
    else:
        import test_our_attack
        from test_our_attack import import_flags, lp_attacks

        flags.FLAGS._flags().clear()
        importlib.reload(test_our_attack)
        script_name = "test_our_attack"

    import_flags(attack)
    norm, attack_klass = lp_attacks[attack]

    num_images = 1000
    batch_size = 250
    attack_grid_args = {
        "num_batches": [num_images // batch_size],
        "batch_size": [batch_size],
        "seed": [seed],
        "attack": [attack],
        "attack_loss": ["log", "cw"],
        "attack_iterations": [500],
        "attack_simultaneous_updates": [True],
        "attack_primal_lr": [1e-1],
        "attack_dual_opt": ["sgd"],
        "attack_dual_opt_kwargs": ["{}"],
        "attack_dual_lr": [1e-1],
        "attack_dual_ema": [False],
        "attack_loop_number_restarts": [10],
        "attack_loop_finetune": [True],
        "attack_loop_r0_sampling_algorithm": ["uniform"],
        "attack_loop_r0_sampling_epsilon": [0.5],
        "attack_loop_r0_ods_init": [False],
        "attack_loop_multitargeted": [False],
        "attack_loop_c0_initial_const": [0.1, 0.01],
        "attack_save": [False],
    }
    if epsilon is not None:
        attack_grid_args["attack_epsilon"] = [epsilon]

    if issubclass(attack_klass, ProximalPrimalDualGradientAttack):
        attack_grid_args.update({
            "attack_primal_opt": ["adam"],
            "attack_primal_opt_kwargs": ["{}"],
            "attack_accelerated": [False],
            "attack_momentum": [0.9],
            "attack_adaptive_momentum": [False],
        })
    else:
        attack_grid_args.update({
            "attack_primal_opt": ["adam"],
            "attack_primal_opt_kwargs": ["{}"],
        })

    if norm == "li":
        attack_grid_args.update(
            {"attack_gradient_preprocessing": [False, True]})

    if attack == "l1g":
        attack_grid_args.update({"attack_hard_threshold": [False, True]})

    if norm == "l0":
        attack_grid_args.update({
            "attack_operator": ["l2/3"],
            "attack_has_ecc": [False],
        })

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/our_{norm}"
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        for attack_arg_value in itertools.product(*attack_grid_args.values()):
            attack_args = dict(zip(attack_arg_names, attack_arg_value))
            attack_args.update({
                "load_from": model,
                "working_dir": working_dir,
            })
            if (attack_args["attack_loop_r0_ods_init"]
                    and attack_args["attack_loop_multitargeted"]):
                continue
            for lr, decay_factor, dlr_decay_factor, lr_decay in itertools.product(
                [0.1], [0.01], [0.1], [True]):
                min_lr = round(lr * decay_factor, 6)
                dlr = attack_args["attack_dual_lr"]
                min_dlr = round(dlr * dlr_decay_factor, 6)
                if lr_decay and min_lr < lr:
                    lr_config = {
                        "schedule": "exp",
                        "config": {
                            **ExpDecay(
                                initial_learning_rate=lr,
                                minimal_learning_rate=min_lr,
                                decay_steps=attack_args["attack_iterations"],
                            ).get_config()
                        },
                    }
                    dlr_config = {
                        "schedule": "linear",
                        "config": {
                            **ExpDecay(
                                initial_learning_rate=dlr,
                                minimal_learning_rate=min_dlr,
                                decay_steps=attack_args["attack_iterations"],
                            ).get_config()
                        },
                    }
                else:
                    lr_config = {
                        "schedule": "constant",
                        "config": {
                            **ConstantDecay(lr).get_config()
                        },
                    }
                    dlr_config = {
                        "schedule": "constant",
                        "config": {
                            **ConstantDecay(learning_rate=dlr).get_config()
                        },
                    }
                if lr_decay:
                    finetune_lr_config = {
                        "schedule": "exp",
                        "config": {
                            **ExpDecay(
                                initial_learning_rate=min_lr,
                                minimal_learning_rate=round(
                                    min_lr * decay_factor, 8),
                                decay_steps=attack_args["attack_iterations"],
                            ).get_config()
                        },
                    }
                    finetune_dlr_config = {
                        "schedule": "linear",
                        "config": {
                            **ExpDecay(
                                initial_learning_rate=min_dlr,
                                minimal_learning_rate=round(
                                    min_dlr * dlr_decay_factor, 8),
                                decay_steps=attack_args["attack_iterations"],
                            ).get_config()
                        },
                    }
                else:
                    finetune_lr_config = {
                        "schedule": "constant",
                        "config": {
                            **ConstantDecay(learning_rate=min_lr).get_config()
                        },
                    }
                    finetune_dlr_config = {
                        "schedule": "constant",
                        "config": {
                            **ConstantDecay(learning_rate=min_dlr).get_config(
                            )
                        },
                    }
                attack_args.update({
                    "attack_loop_lr_config":
                    lr_config,
                    "attack_loop_finetune_lr_config":
                    finetune_lr_config,
                    "attack_loop_dual_lr_config":
                    dlr_config,
                    "attack_loop_finetune_dual_lr_config":
                    finetune_dlr_config,
                })
                base_name = f"cifar10_{type}"
                name = format_name(base_name, attack_args) + "_"
                attack_args["name"] = name
                if name in p or name in existing_names:
                    continue
                existing_names.append(name)
                print(generate_test_optimizer(script_name, **attack_args))


@count_number_of_lines
@cleanflags
def pgd_config(norm, seed=123):
    import test_pgd
    from test_pgd import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_pgd)
    import_flags(norm)

    num_images = 1000
    batch_size = 500

    attack_grid_args = {
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'seed': [seed],
        'norm': [norm],
        'attack_loss': ["ce", "cw"],
        'attack_nb_iter': [500],
        'attack_nb_restarts': [1]
    }
    if norm == 'l1':
        attack_grid_args.update({
            'attack_grad_sparsity': [95, 99]
        })

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/pgd"
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        for attack_arg_value in itertools.product(*attack_grid_args.values()):
            attack_args = dict(zip(attack_arg_names, attack_arg_value))
            attack_args.update({
                'load_from': model,
                'working_dir': working_dir,
            })
            for eps, eps_scale in itertools.product(
                    test_model_thresholds[type][norm],
                [1, 2, 5, 10, 25, 50, 100]):
                attack_args.update({
                    'attack_eps': eps,
                    'attack_eps_iter': eps / eps_scale
                })
                name = f"""cifar10_pgd_{type}_{norm}_{attack_args['attack_loss']}_
n{attack_args['attack_nb_iter']}_N{attack_args['attack_nb_restarts']}_
eps{eps}_epss{eps_scale}_""".replace("\n", "")
                if norm == 'l1':
                    name = f"{name}s{attack_args['attack_grad_sparsity']}_"
                attack_args['name'] = name
                if name in p or name in existing_names:
                    continue
                existing_names.append(name)
                print(generate_test_optimizer('test_pgd', **attack_args))


@cleanflags
def pgd_custom_config(norm, top_k=1, seed=123):
    """Generate config for PGD with 10, 100 restarts based on the results with 1
    restart"""
    import test_pgd
    from test_pgd import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_pgd)
    import_flags(norm)

    num_images = 1000
    batch_size = 500
    default_args = {
        "norm": norm,
        "num_batches": num_images // batch_size,
        "batch_size": batch_size,
        "seed": seed,
    }
    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/pgd/"
        default_args.update({
            'load_from': model,
            'working_dir': working_dir,
        })
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        df_all = load_logs(working_dir)
        for eps in test_model_thresholds[type][norm]:
            df = df_all.copy()
            df = df[df.attack_eps == eps]
            df = df[df.attack_nb_restarts == 1]
            df = df[df.attack_nb_iter == 500]
            acc_col = f"acc_{norm}_{format_float(eps)}"
            df = df.sort_values(acc_col)
            lowest_acc = df.head(1)[acc_col].item()
            i = 0
            for index, df_row in df.iterrows():
                # select top-k attack parameters
                if df_row.at[acc_col] > lowest_acc + 0.01 or i >= top_k:
                    break
                else:
                    attack_args = default_args.copy()
                    for col in df.columns:
                        if "attack_" in col:
                            attack_args[col] = df_row.at[col]
                    eps_scale = int(
                        round(eps / attack_args["attack_eps_iter"], 2))
                    i += 1
                    for loss, n_restarts in itertools.product(["cw", "ce"], [10, 100]):
                        attack_args.update({
                            'attack_nb_restarts': n_restarts,
                            'attack_loss': loss
                        })
                        name = f"""cifar10_pgd_{type}_{norm}_{attack_args['attack_loss']}_
n{attack_args['attack_nb_iter']}_N{attack_args['attack_nb_restarts']}_
eps{eps}_epss{eps_scale}_""".replace("\n", "")
                        if norm == 'l1':
                            name = f"{name}s{attack_args['attack_grad_sparsity']}_"
                        attack_args['name'] = name
                        if name in p or name in existing_names:
                            continue
                        existing_names.append(name)
                        print(
                            generate_test_optimizer('test_pgd', **attack_args))


@count_number_of_lines
@cleanflags
def daa_config(seed=123):
    import test_daa
    from test_daa import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_daa)
    import_flags("blob")

    num_images = 1000
    batch_size = 200
    norm = "li"
    attack_grid_args = {
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'seed': [seed],
        'attack_loss_fn': ["xent", "cw"],
        'attack_nb_iter': [500],
        'attack_nb_restarts': [1],
        'method': ["blob"]
    }

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/daa"
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        for attack_arg_value in itertools.product(*attack_grid_args.values()):
            attack_args = dict(zip(attack_arg_names, attack_arg_value))
            attack_args.update({
                'load_from': model,
                'working_dir': working_dir,
            })
            for eps, eps_scale in itertools.product(
                    test_model_thresholds[type][norm],
                [1, 2, 5, 10, 25, 50, 100]):
                attack_args.update({
                    'attack_eps': eps,
                    'attack_eps_iter': eps / eps_scale
                })
                name = f"""cifar10_daa_{type}_{norm}_
{attack_args['attack_loss_fn']}_{attack_args['method']}_
n{attack_args['attack_nb_iter']}_N{attack_args['attack_nb_restarts']}_
eps{eps}_epss{eps_scale}_""".replace("\n", "")
                attack_args['name'] = name
                if name in p or name in existing_names:
                    continue
                existing_names.append(name)
                print(generate_test_optimizer('test_daa', **attack_args))


@cleanflags
def daa_custom_config(top_k=1, seed=123):
    """Generate config for DAA with 10, 100 restarts based on the results with 1
    restart"""
    import test_daa
    from test_daa import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_daa)
    import_flags("blob")

    num_images = 1000
    batch_size = 200
    norm = "li"
    default_args = {
        "num_batches": num_images // batch_size,
        "batch_size": batch_size,
        "seed": seed,
    }
    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/daa/"
        default_args.update({
            'load_from': model,
            'working_dir': working_dir,
        })
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        df_all = load_logs(working_dir)
        for eps in test_model_thresholds[type][norm]:
            df = df_all.copy()
            df = df[df.attack_eps == eps]
            df = df[df.attack_nb_restarts == 1]
            df = df[df.attack_nb_iter == 500]
            df = df.sort_values("acc_adv")
            lowest_acc = df.head(1).acc_adv.item()
            i = 0
            for index, df_row in df.iterrows():
                # select top-k attack parameters
                if df_row.at['acc_adv'] > lowest_acc + 0.01 or i >= top_k:
                    break
                else:
                    attack_args = default_args.copy()
                    for col in df.columns:
                        if "attack_" in col or col == 'method':
                            attack_args[col] = df_row.at[col]
                    eps_scale = int(
                        round(eps / attack_args["attack_eps_iter"], 2))
                    i += 1
                    for loss, n_restarts in itertools.product(['xent', 'cw'], [10, 100]):
                        attack_args.update({
                            'attack_nb_restarts': n_restarts,
                            'attack_loss_fn': loss
                        })
                        name = f"""cifar10_daa_{type}_{norm}_
{attack_args['attack_loss_fn']}_{attack_args['method']}_
n{attack_args['attack_nb_iter']}_N{attack_args['attack_nb_restarts']}_
eps{eps}_epss{eps_scale}_""".replace("\n", "")
                        attack_args['name'] = name
                        if name in p or name in existing_names:
                            continue
                        existing_names.append(name)
                        print(
                            generate_test_optimizer('test_daa', **attack_args))


@cleanflags
def fab_config(norm, seed=123):
    from lib.fab import FABAttack

    import test_fab

    flags.FLAGS._flags().clear()
    importlib.reload(test_fab)
    import_klass_annotations_as_flags(FABAttack, "attack_")

    num_images = 1000
    batch_size = 250
    attack_args = {
        "attack_norm": norm,
        "num_batches": num_images // batch_size,
        "batch_size": batch_size,
        "seed": seed,
    }

    existing_names = []
    for model, n_iter, n_restarts in itertools.product(models, [100],
                                                       [1, 10, 100]):
        # default params for cifar10
        # see page 12: https://openreview.net/pdf?id=HJlzxgBtwH
        alpha_max = 0.1
        eta = 1.05
        beta = 0.9
        eps = {
            'plain': {'li': 0.0, 'l2':  0.5, 'l1': 10.0},
            'linf':  {'li': 0.02, 'l2': 4.0, 'l1': 10.0},
            'l2':    {'li': 0.02, 'l2': 4.0, 'l1': 10.0}
        }

        # params
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/fab"
        attack_args.update({
            "attack_n_iter": n_iter,
            "attack_n_restarts": n_restarts,
            "attack_alpha_max": alpha_max,
            "attack_eta": eta,
            "attack_beta": beta,
            "attack_eps": eps[type][norm],
            "working_dir": working_dir,
            "load_from": model,
        })
        name = f"cifar10_fab_{type}_{norm}_n{n_iter}_N{n_restarts}_"
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer("test_fab", **attack_args))


@cleanflags
def cleverhans_config(norm, attack, seed=123):
    """Cleverhans attacks config"""
    import test_cleverhans
    from test_cleverhans import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_cleverhans)
    import_flags(norm, attack)

    num_images = 1000
    batch_size = 500
    attack_grid_args = {
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'load_from': models,
        'attack': [attack],
        'norm': [norm],
        'seed': [seed]
    }
    if attack == 'cw':
        # default params
        attack_grid_args.update({
            'attack_max_iterations': [10000],
            'attack_learning_rate': [0.01],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_abort_early': [False],
            'attack_batch_size': [batch_size]
        })
        name_fn = lambda: f"cifar10_{type}_{attack}_cleverhans_n{attack_args['attack_max_iterations']}_lr{attack_args['attack_learning_rate']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_max_iterations': [1000, 10000],
            'attack_learning_rate': [0.01],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_beta': [0.05],
            'attack_abort_early': [False],
            'attack_batch_size': [batch_size]
        })
        name_fn = lambda: f"cifar10_{type}_{attack}_cleverhans_n{attack_args['attack_max_iterations']}_b{attack_args['attack_beta']}_C{attack_args['attack_initial_const']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/{attack}"
        attack_args = dict(zip(attack_arg_names, attack_arg_value))
        attack_args.update({
            'working_dir': working_dir,
        })
        name = name_fn()
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_cleverhans', **attack_args))


@cleanflags
def foolbox_config(norm, attack, seed=123):
    """Foobox attacks"""
    import test_foolbox
    from test_foolbox import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_foolbox)
    import_flags(norm, attack)

    num_images = 1000
    batch_size = 500
    attack_grid_args = {
        "num_batches": [num_images // batch_size],
        "batch_size": [batch_size],
        "load_from": models,
        "attack": [attack],
        "norm": [norm],
        "seed": [seed],
    }
    if attack == "df":
        # default params
        attack_grid_args.update({
            "attack_steps": [50, 100, 1000],
            "attack_overshoot": [0.02],
            "attack_candidates": [10],
        })
        name_fn = (
            lambda:
            f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_os{attack_args['attack_overshoot']}_"
        )
    elif attack == "cw":
        # default params
        attack_grid_args.update({
            "attack_steps": [10000],
            "attack_stepsize": [0.01],
            "attack_initial_const": [0.01],
            "attack_binary_search_steps": [9],
            "attack_abort_early": [False],
        })
        name_fn = (
            lambda:
            f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_C{attack_args['attack_initial_const']}_"
        )
    elif attack == "newton":
        # default params
        attack_grid_args.update({
            "attack_steps": [1000],
            "attack_stepsize": [0.01],
        })
        name_fn = (
            lambda:
            f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_"
        )
    elif attack == "ead":
        # default params
        attack_grid_args.update({
            "attack_steps": [1000, 10000],
            "attack_initial_const": [0.01],
            "attack_binary_search_steps": [9],
            "attack_decision_rule": ["L1"],
            "attack_regularization": [0.05],
            "attack_abort_early": [False],
        })
        name_fn = (
            lambda:
            f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_b{attack_args['attack_regularization']}_C{attack_args['attack_initial_const']}_"
        )
    elif attack == "ddn":
        # default params for cifar10
        # see: http://openaccess.thecvf.com/content_CVPR_2019/papers/Rony_Decoupling_Direction_and_Norm_for_Efficient_Gradient-Based_L2_Adversarial_Attacks_CVPR_2019_paper.pdf
        attack_grid_args.update({
            "attack_steps": [1000],
            "attack_init_epsilon": [1.0, 0.1],
            "attack_gamma": [0.1, 0.05, 0.01],
        })
        name_fn = (
            lambda:
            f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_eps{attack_args['attack_init_epsilon']}_gamma{attack_args['attack_gamma']}_"
        )

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index("load_from")]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/{attack}"
        attack_args = dict(zip(attack_arg_names, attack_arg_value))
        attack_args.update({
            "working_dir": working_dir,
        })
        name = name_fn()
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer("test_foolbox", **attack_args))


@cleanflags
def bethge_config(norm, seed=123):
    import test_bethge
    from test_bethge import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_bethge)
    import_flags(norm)

    num_images = 1000
    batch_size = 250
    attack_args = {
        "norm": norm,
        "num_batches": num_images // batch_size,
        "batch_size": batch_size,
        "seed": seed,
    }

    existing_names = []
    for model, steps, lr, num_decay in itertools.product(
            models, [1000], [1.0, 0.1, 0.01], [20, 100]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/bethge"
        attack_args.update({
            "norm": norm,
            "load_from": model,
            "working_dir": working_dir,
            "attack_steps": steps,
            "attack_lr": lr,
            "attack_lr_num_decay": num_decay,
        })
        name = f"cifar10_bethge_{type}_{norm}_n{steps}_lr{lr}_nd{num_decay}_"
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer("test_bethge", **attack_args))


@cleanflags
def deepfool_config(norm, seed=123):
    import test_deepfool

    flags.FLAGS._flags().clear()
    importlib.reload(test_deepfool)

    assert norm in ['l2', 'li']
    num_images = 1000
    batch_size = 500
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'norm': norm,
        'seed': seed
    }

    existing_names = []
    for model, max_iter in itertools.product(models, [50, 100, 1000]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/df"
        attack_args.update({
            "load_from": model,
            "working_dir": working_dir,
            "attack_overshoot": 0.02,
            "attack_max_iter": max_iter,
        })
        name = f"cifar10_{type}_df_orig_n{attack_args['attack_max_iter']}_os{attack_args['attack_overshoot']}_"
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer("test_deepfool", **attack_args))


@cleanflags
def sparsefool_config(seed=123):
    import test_sparsefool

    flags.FLAGS._flags().clear()
    importlib.reload(test_sparsefool)

    norm = 'l1'
    num_images = 1000
    batch_size = 500
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model, lambda_ in itertools.product(models, [1.0, 2.0, 3.0]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/sparsefool"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_epsilon': 0.02,
            'attack_max_iter': 20,
            'attack_lambda_': lambda_,
        })
        name = f"cifar10_sparsefool_{type}_{norm}_l{lambda_}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_sparsefool', **attack_args))


@cleanflags
def cornersearch_config(seed=123):
    import test_cornersearch

    flags.FLAGS._flags().clear()
    importlib.reload(test_cornersearch)

    norm = "l0"
    num_images = 1000
    batch_size = 500
    attack_args = {
        "num_batches": num_images // batch_size,
        "batch_size": batch_size,
        "seed": seed,
    }

    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/cornersearch"
        attack_args.update({
            "load_from": model,
            "working_dir": working_dir,
            "attack_sparsity": 32 ** 2
        })
        name = f"cifar10_cs_{type}_{norm}_"
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer("test_cornersearch", **attack_args))


@cleanflags
def art_config(norm, attack, seed=123):
    """IBM art toolbox attacks"""
    import test_art
    from test_art import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_art)
    import_flags(norm, attack)

    num_images = 1000
    batch_size = 250
    attack_grid_args = {
        'num_batches':
        [num_images // batch_size],
        'batch_size':
        [batch_size],
        'attack_batch_size': [batch_size],
        'load_from':
        models,
        'attack': [attack],
        'norm': [norm],
        'seed': [seed]
    }
    if attack == 'df':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [50, 100, 1000],
            'attack_nb_grads': [10],
            'attack_epsilon': [0.02],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_art_n{attack_args['attack_max_iter']}_os{attack_args['attack_epsilon']}_"
    elif attack == 'cw':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [10000],
            'attack_initial_const': [1.0, 0.01],
            'attack_binary_search_steps': [9],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_art_n{attack_args['attack_max_iter']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [1000],
            'attack_initial_const': [1.0, 0.01],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_beta': [0.05],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_art_n{attack_args['attack_max_iter']}_b{attack_args['attack_beta']}_C{attack_args['attack_initial_const']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/{attack}"
        attack_args = dict(zip(attack_arg_names, attack_arg_value))
        attack_args.update({
            'working_dir': working_dir,
        })
        name = name_fn()
        attack_args["name"] = name
        p = [
            s.name[:-1] for s in list(Path(working_dir).glob("*"))
        ]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_art', **attack_args))


@cleanflags
def jsma_config(seed=123):
    num_images = 1000
    batch_size = 100
    norm = "l0"
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model, targets, theta, lib in itertools.product(
            models, ["all", "random", "second"],
            [1.0, 0.1], ["cleverhans", "art"]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/jsma"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_targets': targets,
            'attack_theta': theta,
            'attack_gamma': 1.0,
            'attack_impl': lib
        })
        name = f"cifar10_jsma_{type}_{targets}_t{theta}_g1.0_lib{lib}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_jsma', **attack_args))


@cleanflags
def pixel_attack_config(seed=123):
    num_images = 1000
    batch_size = 100
    norm = "l0"
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model, iters, es in itertools.product(models, [100], [1]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../{basedir}/test_{type}/{norm}/one_pixel"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_iters': iters,
            'attack_es': es,
        })
        for threshold in test_model_thresholds[type]["l0"]:
            attack_args['attack_threshold'] = threshold
            name = f"cifar10_one_pixel_{type}_es{es}_i{iters}_t{threshold}_"
            attack_args['name'] = name
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p or name in existing_names:
                continue
            existing_names.append(name)
            print(
                generate_test_optimizer('test_pixel_attack',
                                        **attack_args))


if __name__ == '__main__':
    # our attacks
    test_our_attack_config("l2g")
    test_our_attack_config("l2")
    test_our_attack_config("li")
    test_our_attack_config("l1")
    test_our_attack_config("l0")
    # li attacks
    deepfool_config("li")
    foolbox_config("li", "df")
    bethge_config("li")
    to_execute_cmds = daa_config()
    if to_execute_cmds == 0:
        daa_custom_config()
    to_execute_cmds = pgd_config("li")
    if to_execute_cmds == 0:
        pgd_custom_config("li")
    fab_config("li")
    # l2 attacks
    deepfool_config("l2")
    foolbox_config("l2", "df")
    art_config("l2", "df")
    foolbox_config("l2", "cw")
    cleverhans_config("l2", "cw")
    foolbox_config("l2", "ddn")
    bethge_config("l2")
    to_execute_cmds = pgd_config("l2")
    if to_execute_cmds == 0:
        pgd_custom_config("l2")
    fab_config("l2")
    # l1 attacks
    sparsefool_config()
    cleverhans_config("l1", "ead")
    foolbox_config("l1", "ead")
    bethge_config("l1")
    to_execute_cmds = pgd_config("l1")
    if to_execute_cmds == 0:
        pgd_custom_config("l1")
    fab_config("l1")
    # l0 attacks
    jsma_config()
    cornersearch_config()
    pixel_attack_config()
    bethge_config("l0")
