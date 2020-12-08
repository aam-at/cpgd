"""This script can be used to generate test configs for the compared attacks.
"""
from __future__ import absolute_import, division, print_function

import ast
import functools
import importlib
import itertools
import subprocess
from pathlib import Path

import numpy as np
from absl import flags
from lib.attack_lp import ProximalPrimalDualGradientAttack
from lib.generate_script import format_name, generate_test_optimizer
from lib.parse_logs import parse_log
from lib.tf_utils import ConstantDecay, ExpDecay, LinearDecay
from lib.utils import (import_func_annotations_as_flags,
                       import_klass_annotations_as_flags)

from config import test_model_thresholds

models = [
    './models/mnist_weights_plain.mat', './models/mnist_weights_linf.mat',
    './models/mnist_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')

FLAGS = flags.FLAGS


def cleanflags(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        flags.FLAGS._flags().clear()
        fn(*args, **kwargs)

    return wrapper


def generate_test_optimizer_lp(**kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', **kwargs)


def test_random(runs=1, master_seed=1):
    existing_names = []
    for model, N, norm, eps, init in itertools.product(
            models, [100], ["l2"], np.linspace(0.05, 0.5, 10),
        ["uniform", "sign"]):
        type = Path(model).stem.split("_")[-1]
        eps = round(eps, 3)
        name = f"mnist_{type}_N{N}_{init}_{eps}_"
        working_dir = f"../results/mnist_10/test_random_{type}_{norm}"
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


@cleanflags
def test_our_attack_config(attack, epsilon=None, seed=123):
    if epsilon is not None:
        import test_our_eps_attack
        from test_our_eps_attack import import_flags, lp_attacks
        flags.FLAGS._flags().clear()
        importlib.reload(test_our_eps_attack)
        script_name = 'test_our_eps_attack'
    else:
        import test_our_attack
        from test_our_attack import import_flags, lp_attacks
        flags.FLAGS._flags().clear()
        importlib.reload(test_our_attack)
        script_name = 'test_our_attack'

    import_flags(attack)
    norm, attack_klass = lp_attacks[attack]

    num_images = 1000
    batch_size = 500
    attack_grid_args = {
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'seed': [seed],
        'attack': [attack],
        'attack_loss': ["cw", "exp", "log", "square", "matsushita", "ce"],
        'attack_iterations': [500],
        'attack_simultaneous_updates': [True, False],
        'attack_primal_lr': [1e-1],
        'attack_dual_opt': ["sgd"],
        'attack_dual_opt_kwargs': ["{}"],
        'attack_dual_lr': [1e-1],
        'attack_dual_ema': [True, False],
        'attack_loop_number_restarts': [1],
        'attack_loop_finetune': [True, False],
        'attack_loop_r0_sampling_algorithm': ['uniform'],
        'attack_loop_r0_sampling_epsilon': [0.0, 0.5],
        'attack_loop_r0_ods_init': [True, False],
        'attack_loop_multitargeted': [True, False],
        'attack_loop_c0_initial_const': [1.0, 0.1, 0.01],
        'attack_save': [False]
    }
    if epsilon is not None:
        attack_grid_args['attack_epsilon'] = [epsilon]

    if issubclass(attack_klass, ProximalPrimalDualGradientAttack):
        attack_grid_args.update({
            'attack_primal_opt': ["sgd"],
            'attack_primal_opt_kwargs': ["{}"],
            'attack_accelerated': [True, False],
            'attack_momentum': [0.9],
            'attack_adaptive_momentum': [True, False]
        })
    else:
        attack_grid_args.update({
            'attack_primal_opt': ["adam"],
            'attack_primal_opt_kwargs': ["{}"],
        })

    if norm == 'li':
        attack_grid_args.update(
            {'attack_gradient_preprocessing': [False, True]})

    if attack == 'l1g':
        attack_grid_args.update({'attack_hard_threshold': [False, True]})

    if norm == 'l0':
        attack_grid_args.update({
            'attack_operator': ["l0", "l1", "l1/2", "l2/3"],
            'attack_has_ecc': [False, True]
        })

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/our_{norm}"
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        for attack_arg_value in itertools.product(*attack_grid_args.values()):
            attack_args = dict(zip(attack_arg_names, attack_arg_value))
            attack_args.update({
                'load_from': model,
                'working_dir': working_dir,
            })
            if attack_args['attack_loop_r0_ods_init'] and attack_args['attack_loop_multitargeted']:
                continue
            for lr, decay_factor, lr_decay in itertools.product([1.0, 0.5, 0.1, 0.05, 0.01], [0.01], [True, False]):
                min_lr = round(lr * decay_factor, 6)
                dlr = attack_args['attack_dual_lr']
                min_dlr = round(dlr * decay_factor, 6)
                if lr_decay and min_lr < lr:
                    lr_config = {
                        'schedule': 'exp',
                        'config': {
                            **ExpDecay(initial_learning_rate=lr,
                                       minimal_learning_rate=min_lr,
                                       decay_steps=attack_args['attack_iterations']).get_config(
                            )
                        }
                    }
                    dlr_config = {
                        'schedule': 'exp',
                        'config': {
                            **ExpDecay(initial_learning_rate=dlr,
                                       minimal_learning_rate=min_dlr,
                                       decay_steps=attack_args['attack_iterations']).get_config(
                            )
                        }
                    }
                else:
                    lr_config = {
                        'schedule': 'constant',
                        'config': {
                            **ConstantDecay(lr).get_config()
                        }
                    }
                    dlr_config = {
                        'schedule': 'constant',
                        'config': {
                            **ConstantDecay(learning_rate=dlr).get_config()
                        }
                    }
                if lr_decay:
                    finetune_lr_config = {
                        'schedule': 'exp',
                        'config': {
                            **ExpDecay(initial_learning_rate=min_lr,
                                       minimal_learning_rate=round(
                                           min_lr * decay_factor, 8),
                                       decay_steps=attack_args['attack_iterations']).get_config(
                            )
                        }
                    }
                    finetune_dlr_config = {
                        'schedule': 'exp',
                        'config': {
                            **ExpDecay(initial_learning_rate=min_dlr,
                                       minimal_learning_rate=round(
                                           min_dlr * decay_factor, 8),
                                       decay_steps=attack_args['attack_iterations']).get_config(
                            )
                        }
                    }
                else:
                    finetune_lr_config = {
                        'schedule': 'constant',
                        'config': {
                            **ConstantDecay(learning_rate=min_lr).get_config()
                        }
                    }
                    finetune_dlr_config = {
                        'schedule': 'constant',
                        'config': {
                            **ConstantDecay(learning_rate=min_dlr).get_config(
                            )
                        }
                    }
                attack_args.update({
                    'attack_loop_lr_config': lr_config,
                    'attack_loop_finetune_lr_config': finetune_lr_config,
                    'attack_loop_dual_lr_config': dlr_config,
                    'attack_loop_finetune_dual_lr_config': finetune_dlr_config,
                })
                base_name = f"mnist_{type}"
                name = format_name(base_name, attack_args) + '_'
                attack_args["name"] = name
                if name in p or name in existing_names:
                    continue
                existing_names.append(name)
                print(generate_test_optimizer(script_name, **attack_args))


def test_our_attack_config_custom(attack, topk=1, runs=1, master_seed=1):
    import test_our_attack
    from test_our_attack import import_flags, lp_attacks

    flags.FLAGS._flags().clear()
    importlib.reload(test_our_attack)
    import_flags(attack)
    norm, attack_klass = lp_attacks[attack]
    # import args
    defined_flags = flags.FLAGS._flags().keys()
    test_params = [
        flag for flag in defined_flags if flag.startswith("attack")
        if flag not in ['attack_simultaneous_updates']
    ]

    num_images = 1000
    batch_size = 500
    attack_args = {
        'attack': attack,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/our"
        attack_args.update({'load_from': model, 'working_dir': working_dir})

        # parse test log
        df = parse_test_log(Path(working_dir) / f"mnist_{type}_{attack}_*",
                            export_test_params=test_params)
        df = df[df.attack == attack]
        df = df.sort_values(norm)
        df = df[df.name.str.contains("N100")]
        j = 0
        for id, df in df.iterrows():
            attack_args.update(
                {col: df[col]
                 for col in df.keys() if col in test_params})
            # check args
            if issubclass(attack_klass, ProximalPrimalDualGradientAttack):
                if attack_args['attack_accelerated']:
                    continue
            if attack_args['attack_loop_c0_initial_const'] != 0.01:
                continue
            if attack_args['attack_loop_r0_sampling_epsilon'] != 0.5:
                continue

            lr_config = ast.literal_eval(attack_args['attack_loop_lr_config'])
            flr_config = ast.literal_eval(
                attack_args['attack_loop_finetune_lr_config'])
            if lr_config['schedule'] != 'linear':
                continue
            if flr_config['schedule'] != 'linear':
                continue
            if round(lr_config['config']['initial_learning_rate'] /
                     lr_config['config']['minimal_learning_rate']) != 100:
                continue
            if round(flr_config['config']['initial_learning_rate'] /
                     flr_config['config']['minimal_learning_rate']) != 10:
                continue
            attack_args['working_dir'] = f"../results/mnist_cpgd/test_{type}_{norm}"

            # change args
            j += 1
            for upd, R in itertools.product([True], [1, 10, 100]):
                attack_args['attack_simultaneous_updates'] = upd
                attack_args['attack_loop_number_restarts'] = R
                # generate unique name
                base_name = f"mnist_{type}"
                name = format_name(base_name, attack_args) + '_'
                attack_args["name"] = name
                if name in existing_names:
                    continue
                p = [
                    s.name[:-1]
                    for s in list(Path(attack_args['working_dir']).glob("*"))
                ]
                if name in p or j > topk:
                    continue
                existing_names.append(name)
                np.random.seed(master_seed)
                for i in range(runs):
                    seed = np.random.randint(1000)
                    attack_args["seed"] = seed
                    print(
                        generate_test_optimizer('test_optimizer_lp_madry',
                                                **attack_args))


@cleanflags
def pgd_config(norm, seed=123):
    import test_pgd
    from test_pgd import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_pgd)
    import_flags(norm)

    num_images = 1000
    batch_size = 500
    attack_args = {
        'norm': norm,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        for nb_iter, nb_restarts, eps, eps_scale in itertools.product(
            [100, 500], [1, 10, 100], test_model_thresholds[type][norm],
            [1, 2, 5, 10, 25, 50, 100]):
            working_dir = f"../results_mnist/test_{type}/{norm}/pgd"
            attack_args.update({
                'load_from': model,
                'working_dir': working_dir,
                'attack_nb_restarts': nb_restarts,
                'attack_nb_iter': nb_iter,
                'attack_eps': eps,
                'attack_eps_iter': eps / eps_scale
            })
            name = f"mnist_pgd_{type}_{norm}_n{nb_iter}_N{nb_restarts}_eps{eps}_epss{eps_scale}_"
            if norm == 'l1':
                grad_sparsity = 99
                attack_args['attack_grad_sparsity'] = grad_sparsity
                name = f"{name}sp{grad_sparsity}_"
            attack_args['name'] = name
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p or name in existing_names:
                continue
            existing_names.append(name)
            print(generate_test_optimizer('test_pgd', **attack_args))


@cleanflags
def daa_config(seed=123):
    import test_daa
    from test_daa import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_daa)
    import_flags("blob")

    num_images = 1000
    batch_size = 200
    norm = 'li'
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        for nb_iter, nb_restarts, method, eps, eps_scale in itertools.product(
            [200], [1, 50], ['dgf', 'blob'], test_model_thresholds[type][norm],
            [1, 2, 5, 10, 25, 50, 100]):
            working_dir = f"../results_mnist/test_{type}/{norm}/daa"
            attack_args.update({
                'load_from': model,
                'working_dir': working_dir,
                'method': method,
                'attack_nb_restarts': nb_restarts,
                'attack_nb_iter': nb_iter,
                'attack_eps': eps,
                'attack_eps_iter': eps / eps_scale
            })
            name = f"mnist_daa_{method}_{type}_n{nb_iter}_N{nb_restarts}_eps{eps}_epss{eps_scale}_"
            attack_args['name'] = name
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p or name in existing_names:
                continue
            existing_names.append(name)
            print(generate_test_optimizer('test_daa', **attack_args))


@cleanflags
def fab_config(norm, seed=123):
    from lib.fab import FABAttack

    import test_fab

    flags.FLAGS._flags().clear()
    importlib.reload(test_fab)
    import_klass_annotations_as_flags(FABAttack, 'attack_')

    num_images = 1000
    batch_size = 500
    attack_args = {
        'attack_norm': norm,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model, n_iter, n_restarts in itertools.product(models, [100, 500],
                                                       [1, 10, 100]):
        # default params for mnist
        # see: https://openreview.net/pdf?id=HJlzxgBtwH
        alpha_max = 0.1
        eta = 1.05
        beta = 0.9
        eps = {
            'plain': {
                'li': 0.15,
                'l2': 2.0,
                'l1': 40.0
            },
            'linf': {
                'li': 0.3,
                'l2': 2.0,
                'l1': 40.0
            },
            'l2': {
                'li': 0.3,
                'l2': 2.0,
                'l1': 40.0
            }
        }
        eps['madry'] = eps['linf']

        # params
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/fab"
        attack_args.update({
            'attack_n_iter': n_iter,
            'attack_n_restarts': n_restarts,
            'attack_alpha_max': alpha_max,
            'attack_eta': eta,
            'attack_beta': beta,
            'attack_eps': eps[type][norm],
            'working_dir': working_dir,
            'load_from': model
        })
        name = f"mnist_fab_{type}_{norm}_n{n_iter}_N{n_restarts}_"
        attack_args["name"] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_fab', **attack_args))


@cleanflags
def cleverhans_config(norm, attack, seed=123):
    """Cleverhans attacks config
    """
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
        name_fn = lambda: f"mnist_{type}_{attack}_cleverhans_n{attack_args['attack_max_iterations']}_lr{attack_args['attack_learning_rate']}_C{attack_args['attack_initial_const']}_"
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
        name_fn = lambda: f"mnist_{type}_{attack}_cleverhans_n{attack_args['attack_max_iterations']}_b{attack_args['attack_beta']}_C{attack_args['attack_initial_const']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/{attack}"
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
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'load_from': models,
        'attack': [attack],
        'norm': [norm],
        'seed': [seed]
    }
    if attack == 'df':
        # default params
        attack_grid_args.update({
            'attack_steps': [50, 100, 1000],
            'attack_overshoot': [0.02],
            'attack_candidates': [10]
        })
        name_fn = lambda: f"mnist_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_os{attack_args['attack_overshoot']}_"
    elif attack == 'cw':
        # default params
        attack_grid_args.update({
            'attack_steps': [10000],
            'attack_stepsize': [0.01],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_abort_early': [False],
        })
        name_fn = lambda: f"mnist_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'newton':
        # default params
        attack_grid_args.update({
            'attack_steps': [1000],
            'attack_stepsize': [0.01],
        })
        name_fn = lambda: f"mnist_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_steps': [1000, 10000],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_regularization': [0.05],
            'attack_abort_early': [False],
        })
        name_fn = lambda: f"mnist_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_b{attack_args['attack_regularization']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ddn':
        # default params for mnist
        # see: http://openaccess.thecvf.com/content_CVPR_2019/papers/Rony_Decoupling_Direction_and_Norm_for_Efficient_Gradient-Based_L2_Adversarial_Attacks_CVPR_2019_paper.pdf
        attack_grid_args.update({
            'attack_steps': [1000, 10000],
            'attack_init_epsilon': [1.0, 0.1],
            'attack_gamma': [0.1, 0.05, 0.01],
        })
        name_fn = lambda: f"mnist_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_eps{attack_args['attack_init_epsilon']}_gamma{attack_args['attack_gamma']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/{attack}"
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
        print(generate_test_optimizer('test_foolbox', **attack_args))


@cleanflags
def bethge_config(norm, seed=123):
    import test_bethge
    from test_bethge import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_bethge)
    import_flags(norm)

    num_images = 1000
    batch_size = 500
    attack_args = {
        'norm': norm,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model, steps, lr, num_decay in itertools.product(
            models, [1000], [1.0, 0.1, 0.01], [20, 100]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/bethge"
        attack_args.update({
            'norm': norm,
            'load_from': model,
            'working_dir': working_dir,
            'attack_steps': steps,
            'attack_lr': lr,
            'attack_lr_num_decay': num_decay
        })
        name = f"mnist_bethge_{type}_{norm}_n{steps}_lr{lr}_nd{num_decay}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_bethge', **attack_args))


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
        working_dir = f"../results_mnist/test_{type}/{norm}/df"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_overshoot': 0.02,
            'attack_max_iter': max_iter,
        })
        name = f"mnist_df_orig_"
        name = f"mnist_{type}_df_orig_n{attack_args['attack_max_iter']}_os{attack_args['attack_overshoot']}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_deepfool', **attack_args))


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
        working_dir = f"../results_mnist/test_{type}/{norm}/sparsefool"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_epsilon': 0.02,
            'attack_max_iter': 20,
            'attack_lambda_': lambda_,
        })
        name = f"mnist_sf_orig_{type}_{norm}_l{lambda_}_"
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

    norm = 'l0'
    num_images = 1000
    batch_size = 500
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/cornersearch"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_sparsity': 784
        })
        name = f"mnist_cs_{type}_{norm}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_cornersearch', **attack_args))


@cleanflags
def art_config(norm, attack, seed=123):
    """IBM art toolbox attacks"""
    import test_art
    from test_art import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_art)
    import_flags(norm, attack)

    num_images = 1000
    batch_size = 500
    attack_grid_args = {
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'attack_batch_size': [batch_size],
        'load_from': models,
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
        name_fn = lambda: f"mnist_{type}_{attack}_art_n{attack_args['attack_max_iter']}_os{attack_args['attack_epsilon']}_"
    elif attack == 'cw':
        # default params
        if norm == "l2":
            attack_grid_args.update({
                'attack_max_iter': [10000],
                'attack_initial_const': [0.01],
                'attack_binary_search_steps': [9],
            })
            name_fn = lambda: f"mnist_{type}_{attack}_art_n{attack_args['attack_max_iter']}_C{attack_args['attack_initial_const']}_"
        else:
            attack_grid_args.update({
                'attack_max_iter': [1000],
                'attack_eps': [0.3],
            })
            name_fn = lambda: f"mnist_{type}_{attack}_art_n{attack_args['attack_max_iter']}_eps{attack_args['attack_eps']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [1000],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_beta': [0.05],
        })
        name_fn = lambda: f"mnist_{type}_{attack}_art_n{attack_args['attack_max_iter']}_b{attack_args['attack_beta']}_C{attack_args['attack_initial_const']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/{attack}"
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
        print(generate_test_optimizer('test_art', **attack_args))


@cleanflags
def jsma_config(seed=123):
    num_images = 1000
    batch_size = 500
    norm = "l0"
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }
    existing_names = []
    for model, targets, theta, lib in itertools.product(
            models, ["all", "random", "second"], [1.0, 0.1],
        ["cleverhans", "art"]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/jsma"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_targets': targets,
            'attack_theta': theta,
            'attack_gamma': 1.0,
            'attack_impl': lib
        })
        name = f"mnist_jsma_{type}_{targets}_t{theta}_g1.0_lib{lib}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_jsma', **attack_args))


@cleanflags
def pixel_attack_config(seed=123):
    num_images = 1000
    batch_size = 200
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }
    norm = "l0"

    existing_names = []
    for model, iters, es in itertools.product(models, [100], [1]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_mnist/test_{type}/{norm}/one_pixel"
        attack_args.update({
            'load_from': model,
            'working_dir': working_dir,
            'attack_iters': iters,
            'attack_es': es,
        })
        for threshold in test_model_thresholds[type]["l0"]:
            attack_args['attack_threshold'] = threshold
            name = f"mnist_one_pixel_{type}_es{es}_i{iters}_t{threshold}_"
            attack_args['name'] = name
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p or name in existing_names:
                continue
            existing_names.append(name)
            print(
                generate_test_optimizer('test_pixel_attack',
                                        **attack_args))


if __name__ == '__main__':
    # li attacks
    deepfool_config("li")
    foolbox_config("li", "df")
    bethge_config("li")
    daa_config()
    pgd_config("li")
    fab_config("li")
    # l2 attacks
    deepfool_config("l2")
    foolbox_config("l2", "df")
    art_config("l2", "df")
    foolbox_config("l2", "cw")
    cleverhans_config("l2", "cw")
    foolbox_config("l2", "ddn")
    foolbox_config("l2", "newton")
    bethge_config("l2")
    pgd_config("l2")
    fab_config("l2")
    # l1 attacks
    sparsefool_config()
    cleverhans_config("l1", "ead")
    foolbox_config("l1", "ead")
    bethge_config("l1")
    pgd_config("l1")
    fab_config("l1")
    # l0 attacks
    jsma_config()
    cornersearch_config()
    pixel_attack_config()
    bethge_config("l0")
