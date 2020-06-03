from __future__ import absolute_import, division, print_function

import ast
import importlib
import itertools
import subprocess
from pathlib import Path

import numpy as np
from absl import flags

from config import test_model_thresholds
from lib.attack_lp import ProximalGradientOptimizerAttack
from lib.fab import FABAttack
from lib.generate_script import format_name, generate_test_optimizer
from lib.parse_logs import parse_test_log
from lib.utils import ConstantDecay, LinearDecay, import_klass_annotations_as_flags

models = [
    './models/cifar10_weights_plain.mat', './models/cifar10_weights_linf.mat',
    './models/cifar10_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')

FLAGS = flags.FLAGS


def generate_test_random(**kwargs):
    return generate_test_optimizer('test_random', **kwargs)


def generate_test_optimizer_lp(**kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', **kwargs)


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
            print(generate_test_random(**attack_args))


def test_lp_config(attack, runs=1, master_seed=1):
    norm, attack_klass = lp_attacks[attack]
    num_images = {'l0': 1000, 'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 500
    attack_grid_args = {
        'num_batches':
        [num_images // batch_size],
        'batch_size':
        [batch_size],
        'load_from':
        models,
        'attack': [attack],
        'attack_loss': ["cw"],
        'attack_iterations': [500],
        'attack_simultaneous_updates': [True, False],
        'attack_primal_lr': [1e-1],
        'attack_dual_optimizer': ["sgd"],
        'attack_dual_lr': [1e-1],
        'attack_dual_ema': [True],
        'attack_use_proxy_constraint': [False],
        'attack_loop_number_restarts': [10],
        'attack_loop_finetune': [True],
        'attack_loop_r0_sampling_algorithm': ['uniform'],
        'attack_loop_r0_sampling_epsilon': [0.5],
        'attack_loop_c0_initial_const': [1.0, 0.1, 0.01]
    }

    if attack == 'l1g':
        attack_grid_args.update({
            'attack_hard_threshold': [True, False]
        })

    if issubclass(attack_klass, ProximalGradientOptimizerAttack):
        attack_grid_args.update({
            'attack_primal_optimizer': ["sgd"],
            'attack_accelerated': [False],
            'attack_momentum': [0.9],
            'attack_adaptive_momentum': [True, False]
        })
    else:
        attack_grid_args.update({
            'attack_primal_optimizer': ["adam"]
        })

    if norm == 'li':
        attack_grid_args.update({
            'attack_gradient_preprocessing': [True]
        })

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_10/test_{type}_{norm}"
        attack_args = dict(zip(attack_arg_names, attack_arg_value))
        attack_args.update({
            'working_dir': working_dir,
        })
        for lr, decay_factor, lr_decay in itertools.product([0.01, 0.05, 0.1, 0.5], [0.1, 0.01], [True]):
            min_lr = round(lr * decay_factor, 6)
            if lr_decay and min_lr < lr:
                lr_config = {
                    'schedule': 'linear',
                    'config': {
                        **LinearDecay(initial_learning_rate=lr,
                                      minimal_learning_rate=min_lr,
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
            if lr_decay:
                finetune_lr_config = {
                    'schedule': 'linear',
                    'config': {
                        **LinearDecay(initial_learning_rate=min_lr,
                                      minimal_learning_rate=round(
                                          min_lr / 10, 6),
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
            attack_args.update({
                'attack_loop_lr_config':
                lr_config,
                'attack_loop_finetune_lr_config':
                finetune_lr_config
            })
            base_name = f"cifar10_{type}"
            name = format_name(base_name, attack_args) + '_'
            attack_args["name"] = name
            p = [
                s.name[:-1] for s in list(Path(working_dir).glob("*"))
            ]
            if name in p or name in existing_names:
                continue
            existing_names.append(name)
            np.random.seed(master_seed)
            for i in range(runs):
                seed = np.random.randint(1000)
                attack_args["seed"] = seed
                if True:
                    print(generate_test_optimizer_lp(**attack_args))


def test_lp_custom_config(attack, topk=1, runs=1, master_seed=1):
    import test_optimizer_lp_madry
    from test_optimizer_lp_madry import lp_attacks

    flags.FLAGS._flags().clear()
    importlib.reload(test_optimizer_lp_madry)
    assert attack in lp_attacks
    norm, attack_klass = lp_attacks[attack]
    import_klass_annotations_as_flags(attack_klass, 'attack_')
    # import args
    defined_flags = flags.FLAGS._flags().keys()
    test_params = [
        flag for flag in defined_flags if flag.startswith("attack")
        if flag not in ['attack_simultaneous_updates']
    ]

    num_images = {'l0': 1000, 'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 250
    attack_args = {
        'attack': attack,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_10/test_{type}_{norm}"
        attack_args.update({'load_from': model, 'working_dir': working_dir})

        # parse test log
        df = parse_test_log(Path(working_dir) / f"cifar10_{type}_{attack}_*",
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
            if issubclass(attack_klass, ProximalGradientOptimizerAttack):
                if attack_args['attack_accelerated']:
                    continue
            if attack_args['attack_loop_c0_initial_const'] != 0.01:
                continue
            if attack_args['attack_loop_r0_sampling_epsilon'] != 0.5:
                continue
            lr_config = ast.literal_eval(attack_args['attack_loop_lr_config'])
            flr_config = ast.literal_eval(attack_args['attack_loop_finetune_lr_config'])
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
            attack_args['working_dir'] = f"../results/cifar10_final/test_{type}_{norm}"

            # change args
            j += 1
            for upd, r, R in itertools.product([True, False], [0.25], [1, 10, 100]):
                attack_args['attack_simultaneous_updates'] = upd
                attack_args['attack_loop_number_restarts'] = R
                attack_args['attack_loop_r0_sampling_epsilon'] = r
                # generate unique name
                base_name = f"cifar10_{type}"
                name = format_name(base_name, attack_args) + '_'
                attack_args["name"] = name
                if name in existing_names:
                    continue
                p = [s.name[:-1] for s in list(Path(attack_args['working_dir']).glob("*"))]
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


def fab_config(norm, runs=1, master_seed=1):
    flags.FLAGS._flags().clear()
    import_klass_annotations_as_flags(FABAttack, 'attack_')

    num_images = {'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 250
    attack_args = {
        'attack_norm': norm,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for model, n_restarts in itertools.product(models, [1, 10, 100]):
        # default params for cifar10
        # see page 12: https://openreview.net/pdf?id=HJlzxgBtwH
        n_iter = 100
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
        working_dir = f"../results/cifar10_fab/test_{type}_{norm}"
        attack_args.update(
        {
            'attack_n_iter': n_iter,
            'attack_n_restarts': n_restarts,
            'attack_alpha_max': alpha_max,
            'attack_eta': eta,
            'attack_beta': beta,
            'attack_eps': eps[type][norm],
            'working_dir': working_dir,
            'load_from': model
        })
        name = f"cifar10_fab_{type}_{norm}_n{n_iter}_N{n_restarts}_"
        attack_args["name"] = name
        p = [
            s.name[:-1] for s in list(Path(working_dir).glob("*"))
        ]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        np.random.seed(master_seed)
        for i in range(runs):
            seed = np.random.randint(1000)
            attack_args["seed"] = seed
            print(generate_test_optimizer('test_fab', **attack_args))


def foolbox_config(norm, attack, runs=1, master_seed=1):
    import test_foolbox
    from test_foolbox import lp_attacks

    flags.FLAGS._flags().clear()
    importlib.reload(test_foolbox)
    if attack == 'ead':
        flags.DEFINE_string("attack_decision_rule", "L1", "")
    import_klass_annotations_as_flags(lp_attacks[norm][attack], prefix="attack_")

    num_images = {'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 250
    attack_grid_args = {
        'num_batches':
        [num_images // batch_size],
        'batch_size':
        [batch_size],
        'load_from':
        models,
        'attack': [attack],
        'norm': [norm]
    }
    if attack == 'df':
        # default params
        attack_grid_args.update({
            'attack_steps': [100],
            'attack_overshoot': [0.02],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_os{attack_args['attack_overshoot']}_"
    elif attack == 'cw':
        # default params
        attack_grid_args.update({
            'attack_steps': [10000],
            'attack_stepsize': [0.01],
            'attack_initial_const': [1.0, 0.001],
            'attack_binary_search_steps': [9],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_steps': [1000],
            'attack_initial_const': [1.0, 0.001],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_regularization': [0.05],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_b{attack_args['attack_regularization']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ddn':
        # default params for cifar10
        # see: http://openaccess.thecvf.com/content_CVPR_2019/papers/Rony_Decoupling_Direction_and_Norm_for_Efficient_Gradient-Based_L2_Adversarial_Attacks_CVPR_2019_paper.pdf
        attack_grid_args.update({
            'attack_steps': [1000],
            'attack_init_epsilon': [1.0],
            'attack_gamma': [0.05],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_eps{attack_args['attack_init_epsilon']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_{attack}/test_{type}_{norm}"
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
        np.random.seed(master_seed)
        for i in range(runs):
            seed = np.random.randint(1000)
            attack_args["seed"] = seed
            print(generate_test_optimizer('test_foolbox', **attack_args))


def bethge_config(norm, runs=1, master_seed=1):
    import test_bethge_attack
    from test_bethge_attack import lp_attacks

    flags.FLAGS._flags().clear()
    importlib.reload(test_bethge_attack)
    attack_klass = lp_attacks[norm]
    import_klass_annotations_as_flags(attack_klass, 'attack_')

    assert norm in lp_attacks
    num_images = {'l0': 1000, 'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 100
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for model, lr, num_decay in itertools.product(models, [1.0, 0.1, 0.01, 0.001], [20, 100]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_bethge/test_{type}_{norm}"
        attack_args.update({
            'norm': norm,
            'load_from': model,
            'working_dir': working_dir,
            'attack_lr': lr,
            'attack_lr_num_decay': num_decay,
        })
        name = f"cifar10_bethge_{type}_{norm}_lr{lr}_nd{num_decay}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_bethge_attack', **attack_args))


def art_config(norm, attack, runs=1, master_seed=1):
    import test_art
    from test_art import lp_attacks

    flags.FLAGS._flags().clear()
    importlib.reload(test_art)
    import_klass_annotations_as_flags(lp_attacks[norm][attack],
                                      prefix="attack_",
                                      include_kwargs_with_defaults=True)

    num_images = {'li': 1000, 'l1': 1000, 'l2': 500}[norm]
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
        'norm': [norm]
    }
    if attack == 'df':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [100],
            'attack_nb_grads': [10],
            'attack_epsilon': [0.02],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_art_n{attack_args['attack_max_iter']}_os{attack_args['attack_epsilon']}_"
    elif attack == 'cw':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [10000],
            'attack_initial_const': [1.0, 0.001],
            'attack_binary_search_steps': [9],
        })
        name_fn = lambda : f"cifar10_{type}_{attack}_art_n{attack_args['attack_max_iter']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [1000],
            'attack_initial_const': [1.0, 0.001],
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
        working_dir = f"../results/cifar10_{attack}/test_{type}_{norm}"
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
        np.random.seed(master_seed)
        for i in range(runs):
            seed = np.random.randint(1000)
            attack_args["seed"] = seed
            print(generate_test_optimizer('test_art', **attack_args))


def jsma_config(runs=1, master_seed=1):
    num_images = 1000
    batch_size = 100
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for model, targets, theta, lib in itertools.product(
            models, ["all", "random", "second"],
            [1.0, 0.1], ["cleverhans", "art"]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_jsma/test_{type}"
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


def one_pixel_attack_config(runs=1, master_seed=1):
    num_images = 1000
    batch_size = 100
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for model, iters, es in itertools.product(models, [100], [1]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_one_pixel/test_{type}"
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
                generate_test_optimizer('test_one_pixel_attack',
                                        **attack_args))


if __name__ == '__main__':
    pass
