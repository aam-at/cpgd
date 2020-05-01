from __future__ import absolute_import, division, print_function

import importlib
import itertools
import subprocess
from pathlib import Path

import numpy as np
from absl import flags

import test_optimizer_lp_madry
from lib.generate_script import format_name, generate_test_optimizer
from lib.parse_logs import parse_test_optimizer_log
from lib.utils import ConstantDecay, LinearDecay, import_klass_kwargs_as_flags
from test_optimizer_lp_madry import lp_attacks

models = [
    './models/mnist_weights_plain.mat', './models/mnist_weights_linf.mat',
    './models/mnist_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')


FLAGS = flags.FLAGS

def generate_test_optimizer_lp(**kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', **kwargs)


def generate_test_bethge_lp(**kwargs):
    return generate_test_optimizer('test_bethge_attack', **kwargs)


def test_lp_config(attack, runs=1, master_seed=1):
    norm, _ = lp_attacks[attack]
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
        'attack_primal_lr': [1e-1],
        'attack_dual_optimizer': ["sgd"],
        'attack_dual_lr': [1e-1],
        'attack_dual_ema': [True, False],
        'attack_use_proxy_constraint': [False],
        'attack_loop_number_restarts': [1],
        'attack_loop_finetune': [True],
        'attack_loop_r0_sampling_algorithm': ['uniform'],
        'attack_loop_r0_sampling_epsilon': [0.5],
        'attack_loop_c0_initial_const': [1.0, 0.1, 0.01]
    }

    if attack != 'l2g':
        attack_grid_args.update({
            'attack_primal_optimizer': ["sgd"],
            'attack_accelerated': [True, False],
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

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist_10/test_{type}_{norm}"
        attack_args = dict(zip(attack_arg_names, attack_arg_value))
        attack_args.update({
            'working_dir': working_dir,
        })
        for lr, decay_factor, lr_decay in itertools.product([0.05, 0.1, 0.5, 1.0], [1, 0.1, 0.01], [True, False]):
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
            base_name = f"mnist_{type}"
            name = format_name(base_name, attack_args) + '_'
            attack_args["name"] = name
            p = [
                s.name[:-1] for s in list(Path(working_dir).glob("*"))
            ]
            if name in p:
                continue
            np.random.seed(master_seed)
            for i in range(runs):
                seed = np.random.randint(1000)
                attack_args["seed"] = seed
                if True:
                    print(generate_test_optimizer_lp(**attack_args))


def test_lp_custom_config(attack, runs=1, master_seed=1):
    norm, _ = lp_attacks[attack]
    num_images = {'l0': 1000, 'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 500
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist_10/test_{type}_{norm}"
        script_module = importlib.import_module("test_optimizer_lp_madry")
        defined_flags = script_module.FLAGS._flags().keys()
        export_test_params=[
            flag for flag in defined_flags if flag.startswith("attack_")]
        df = parse_test_optimizer_log(
            working_dir,
            export_test_params=export_test_params)
        df = df.sort_values(norm)
        df = df[df.name.str.contains("N10")].head(1)
        for id, df in df.iterrows():
            attack_args = {col: df[col] for col in df.keys() if col in export_test_params}
            attack_args.update({
                'attack': attack,
                'num_batches': num_images // batch_size,
                'batch_size': batch_size
            })
            # change args
            attack_args['attack_loop_number_restarts'] = 10

            # generate unique name
            base_name = f"mnist_{type}"
            name = format_name(base_name, attack_args) + '_'
            attack_args["name"] = name
            p = [
                s.name[:-1] for s in list(Path(working_dir).glob("*"))
            ]
            if name in p:
                continue
            np.random.seed(master_seed)
            for i in range(runs):
                seed = np.random.randint(1000)
                attack_args["seed"] = seed
                if True:
                    print(generate_test_optimizer_lp(**attack_args))


def test_bethge_config(norm, runs=1, master_seed=1):
    assert norm in ['l0', 'li', 'l1', 'l2']
    num_images = {'l0': 10, 'li': 10, 'l1': 10, 'l2': 5}[norm]
    attack_args = {'norm': norm, 'num_images': num_images, 'seed': 1}
    name = "mnist_bethge_"

    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist_bethge/test_{norm}_{type}"
        attack_args0 = attack_args.copy()
        attack_args0.update({
            'name': name,
            'norm': norm,
            'load_from': model,
            'working_dir': working_dir
        })
        print(generate_test_bethge_lp(**attack_args0))


if __name__ == '__main__':
    for attack in lp_attacks:
        norm, _ = lp_attacks[attack]
        if norm not in ['l0', 'l1', 'l2', 'li']:
            continue
        flags.FLAGS._flags().clear()
        importlib.reload(test_optimizer_lp_madry)
        _, attack_klass = lp_attacks[attack]
        import_klass_kwargs_as_flags(attack_klass, 'attack_')
        test_lp_config(attack)
