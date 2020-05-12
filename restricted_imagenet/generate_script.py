from __future__ import absolute_import, division, print_function

import importlib
import itertools
import os
import subprocess
from pathlib import Path

import numpy as np
from absl import flags

from lib.attack_lp import ProximalGradientOptimizerAttack
from lib.generate_script import format_name, generate_test_optimizer
from lib.parse_logs import parse_test_log
from lib.utils import ConstantDecay, LinearDecay, import_klass_kwargs_as_flags

models = {
    'plain': './models/train_224_nat_slim',
    'linf': './models/train_224_robust_eps_0.005_lp_inf_slim',
    'l2': './models/train_224_robust_eps_1.0_lp_2_slim'
}
hostname = subprocess.getoutput('hostname')

FLAGS = flags.FLAGS


def generate_test_random(**kwargs):
    return generate_test_optimizer('test_random', **kwargs)


def generate_test_optimizer_lp(**kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', **kwargs)


def generate_test_bethge_lp(**kwargs):
    return generate_test_optimizer('test_bethge_attack', **kwargs)


def generate_test_jsma(**kwargs):
    return generate_test_optimizer('test_jsma', **kwargs)

def test_random(runs=1, master_seed=1):
    existing_names = []
    for model, N, norm, eps, init in itertools.product(
            models.keys(), [10], ["l2"], np.linspace(0.05, 0.5, 10),
        ["uniform", "sign"]):
        type = Path(model).stem.split("_")[-1]
        eps = round(eps, 3)
        name = f"imagenet_{type}_N{N}_{init}_{eps}_"
        working_dir = f"../results/imagenet/test_random_{type}_{norm}"
        attack_args = {
            'working_dir': working_dir,
            'data_dir': os.environ['IMAGENET_DIR'],
            'load_from': models[model],
            'num_batches': 5,
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
    num_images = {'l0': 500, 'li': 500, 'l1': 500, 'l2': 500}[norm]
    batch_size = 50
    attack_grid_args = {
        'num_batches': [num_images // batch_size],
        'batch_size': [batch_size],
        'attack': [attack],
        'attack_loss': ["cw"],
        'attack_iterations': [500],
        'attack_primal_lr': [1e-1],
        'attack_dual_optimizer': ["sgd"],
        'attack_dual_lr': [1e-1],
        'attack_dual_ema': [True],
        'attack_use_proxy_constraint': [False],
        'attack_loop_number_restarts': [1],
        'attack_loop_finetune': [True],
        'attack_loop_r0_sampling_algorithm': ['uniform'],
        'attack_loop_r0_sampling_epsilon': [0.1, 0.3, 0.5],
        'attack_loop_c0_initial_const': [0.01]
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
        attack_grid_args.update({'attack_primal_optimizer': ["adam"]})

    if norm == 'li':
        attack_grid_args.update({'attack_gradient_preprocessing': [True]})

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for type in models.keys():
        for attack_arg_value in itertools.product(*attack_grid_args.values()):
            working_dir = f"../results/imagenet_10/test_{type}_{norm}"
            attack_args = dict(zip(attack_arg_names, attack_arg_value))
            attack_args.update({
                'working_dir': working_dir,
                'load_from': models[type]
            })
            for lr, decay_factor, flr_decay_factor, lr_decay in itertools.product(
                [0.005, 0.01, 0.05, 0.1, 0.5], [0.01], [0.1], [True]):
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
                if lr_decay and flr_decay_factor < 1.0:
                    finetune_lr_config = {
                        'schedule': 'linear',
                        'config': {
                            **LinearDecay(initial_learning_rate=min_lr,
                                          minimal_learning_rate=round(
                                              min_lr * flr_decay_factor, 6),
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
                base_name = f"imagenet_{type}"
                name = format_name(base_name, attack_args) + '_'
                attack_args["name"] = name
                p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
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
    norm, _ = lp_attacks[attack]
    num_images = {'l0': 500, 'li': 500, 'l1': 500, 'l2': 500}[norm]
    batch_size = 50
    existing_names = []
    for type in models.keys():
        working_dir = f"../results/imagenet_10/test_{type}_{norm}"
        script_module = importlib.import_module("test_optimizer_lp_madry")
        defined_flags = script_module.FLAGS._flags().keys()
        export_test_params = [
            flag for flag in defined_flags if flag.startswith("attack_")
        ]
        df = parse_test_optimizer_log(Path(working_dir) /
                                      f"imagenet_{type}_{attack}_",
                                      export_test_params=export_test_params)
        if len(df) == 0:
            continue
        df = df[df.name.str.contains("N1")]
        df = df.sort_values(norm)
        j = 0
        for id, df in df.iterrows():
            attack_args = {
                col: df[col]
                for col in df.keys() if col in export_test_params
            }
            attack_args.update({
                'attack': attack,
                'num_batches': num_images // batch_size,
                'batch_size': batch_size,
                'load_from': models[type],
                'working_dir': working_dir
            })
            if attack != 'l2g' and not attack_args[
                    'attack_accelerated'] and not attack_args[
                        'attack_adaptive_momentum']:
                continue
            import ast
            lr_config = ast.literal_eval(attack_args['attack_loop_lr_config'])
            flr_config = ast.literal_eval(attack_args['attack_loop_finetune_lr_config'])
            if lr_config['schedule'] != 'linear':
                continue
            if flr_config['schedule'] != 'linear':
                continue
            if lr_config['config']['initial_learning_rate'] / lr_config['config']['minimal_learning_rate'] < 11:
                continue
            if flr_config['config']['initial_learning_rate'] / lr_config['config']['minimal_learning_rate'] > 11:
                continue
            if attack_args['attack_loop_c0_initial_const'] != 0.01:
                continue
            # change args
            attack_args['attack_loop_number_restarts'] = 10
            attack_args['attack_loop_r0_sampling_epsilon'] = 0.5

            # generate unique name
            base_name = f"imagenet_{type}"
            name = format_name(base_name, attack_args) + '_'
            attack_args["name"] = name
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in existing_names:
                continue
            j += 1
            if name in p or j > topk:
                continue
            existing_names.append(name)
            np.random.seed(master_seed)
            for i in range(runs):
                seed = np.random.randint(1000)
                attack_args["seed"] = seed
                if True:
                    print(generate_test_optimizer_lp(**attack_args))


def test_bethge_config(norm, runs=1, master_seed=1):
    assert norm in ['l0', 'li', 'l1', 'l2']
    num_images = {'l0': 500, 'li': 500, 'l1': 500, 'l2': 500}[norm]
    batch_size = 50
    attack_args = {
        'norm': norm,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for type, l0_pixel in itertools.product(models.keys(), [True, False]):
        working_dir = f"../results/imagenet_bethge/test_{type}_{norm}"
        attack_args.update({
            'norm': norm,
            'load_from': models[type],
            'working_dir': working_dir,
        })
        name = f"imagenet_bethge_{type}_{norm}_"
        if norm == 'l0':
            attack_args["attack_l0_pixel_metric"] = l0_pixel
            name = f"{name}{'pixel_' if l0_pixel else ''}"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_bethge_lp(**attack_args))


def test_jsma_config(runs=1, master_seed=1):
    num_images = 1000
    batch_size = 100
    attack_args = {
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for type, l0_pixel in itertools.product(models.keys(), [True, False]):
        working_dir = f"../results/imagenet_jsma/test_{type}"
        attack_args.update({
            'load_from': models[type],
            'working_dir': working_dir,
            'attack_l0_pixel_metric': l0_pixel,
        })
        name = f"imagenet_jsma_{type}_{targets}_{'pixel_' if l0_pixel else ''}"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_jsma(**attack_args))


if __name__ == '__main__':
    test_random()
    # import test_optimizer_lp_madry
    # from test_optimizer_lp_madry import lp_attacks
    # for attack in lp_attacks:
    #     norm, _ = lp_attacks[attack]
    #     if norm not in ['l0']:
    #         continue
    #     flags.FLAGS._flags().clear()
    #     importlib.reload(test_optimizer_lp_madry)
    #     _, attack_klass = lp_attacks[attack]
    #     import_klass_kwargs_as_flags(attack_klass, 'attack_')
    #     test_lp_config(attack)
    #     # test_lp_custom_config(attack)
