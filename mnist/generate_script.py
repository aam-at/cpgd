from __future__ import absolute_import, division, print_function

import importlib
import itertools
import subprocess
from pathlib import Path
import importlib

import numpy as np

import test_optimizer_lp_madry
from lib.generate_script import generate_test_optimizer
from lib.parse_logs import parse_test_optimizer_log
from lib.utils import ConstantDecay, LinearDecay, import_klass_kwargs_as_flags

from test_optimizer_lp_madry import lp_attacks

models = [
    './models/mnist_weights_plain.mat', './models/mnist_weights_linf.mat',
    './models/mnist_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')


from absl import flags
FLAGS = flags.FLAGS

def generate_test_optimizer_lp(**kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', **kwargs)


def generate_test_bethge_lp(**kwargs):
    return generate_test_optimizer('test_bethge_attack', **kwargs)


def format_name(base_name, attack_args):
    name = f"""{base_name}_{attack_args['attack']}_{attack_args["attack_loss"]}_n{attack_args["attack_iterations"]}
_N{attack_args["attack_loop_number_restarts"]}
"""
    lr_config = attack_args['attack_loop_lr_config']
    if lr_config['schedule'] == 'constant':
        name = f"{name}_lr{lr_config['config']['learning_rate']}"
    elif lr_config['schedule'] == 'linear':
        name = f"""{name}_linear_lr{lr_config['config']['initial_learning_rate']:.2f}_
mlr{lr_config['config']['minimal_learning_rate']:.3f}_
{'finetune' if attack_args['attack_loop_finetune'] else 'nofinetune'}
"""
    name = f"{name}_{attack_args['attack_primal_optimizer']}"
    if 'attack_gradient_preprocessing' in attack_args and attack_args[
            'attack_gradient_preprocessing']:
        name = f"{name}_gprep"
    if 'attack_accelerated' in attack_args and attack_args[
            'attack_accelerated']:
        name = f"{name}_apg_m{attack_args['attack_momentum']}" \
               f"{'_adaptive' if attack_args['attack_adaptive_momentum'] else ''}"
    name = f"{name}_dlr{attack_args['attack_dual_lr']}_d{attack_args['attack_dual_optimizer']}"
    name = f"""{name}_{attack_args['attack_loop_r0_sampling_algorithm']}_
R{attack_args['attack_loop_r0_sampling_epsilon']}_
C{attack_args['attack_loop_c0_initial_const']}_
{'proxy' if attack_args['attack_use_proxy_constraint'] else 'noproxy'}"""
    return name.replace('\n', '')


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
        for lr, lr_decay in itertools.product([0.05, 0.1, 0.5, 1.0], [1, 0.1]):
            min_lr = lr * lr_decay
            if lr == min_lr:
                lr_config = {
                    'schedule': 'constant',
                    'config': {
                        **ConstantDecay(lr).get_config()
                    }
                }
            else:
                assert lr_decay < 1
                lr_config = {
                    'schedule': 'linear',
                    'config': {
                        **LinearDecay(initial_learning_rate=lr,
                                      minimal_learning_rate=min_lr,
                                      decay_steps=attack_args['attack_iterations']).get_config(
                        )
                    }
                }
            finetune_lr_config = {
                'schedule': 'linear',
                'config': {
                    **LinearDecay(initial_learning_rate=min_lr,
                                  minimal_learning_rate=min_lr / 10,
                                  decay_steps=attack_args['attack_iterations']).get_config(
                    )
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


def test_lp_custom_config(norm, runs=1, master_seed=1):
    for model in models:
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist_10/test_{norm}_{type}"
        script_module = importlib.import_module("test_optimizer_lp_madry")
        defined_flags = script_module.FLAGS._flags().keys()
        export_test_params=[
            flag for flag in defined_flags if flag.startswith("attack_")]
        if norm == 'li':
            export_test_params.append("attack_use_sign")
        export_test_params.extend(['seed', 'num_batches', 'batch_size', 'norm'])
        df = parse_test_optimizer_log(
            working_dir,
            export_test_params=export_test_params)
        df = df[getattr(df, f"acc_{norm}") == 0]

        def get_group_name(name):
            return "_".join(name.split("_")[5:6])

        df['group_name'] = [get_group_name(n) for n in df['name']]
        df = df.sort_values(norm).groupby('group_name', axis=0).head(50)
        df = df[df.name.str.contains("_0k")]
        for id, df in df.iterrows():
            attack_args = {
                'norm': norm,
                'num_batches': df.num_batches * df.batch_size // 500,
                'batch_size': 500,
                'seed': df.seed
            }
            # load args
            iterations = df.attack_iterations
            max_iterations = 10 * df.attack_max_iterations
            loss = df.attack_loss
            optimizer = df.attack_primal_optimizer
            accelerated = df.attack_accelerated
            gradient_normalize = df.attack_gradient_normalize
            name = "mnist_lin_lambd_"
            name = name + f"{norm}_3_{type}_{iterations}_{max_iterations // 1000}k_"
            name = name + f"{optimizer if not accelerated else 'apg_sgd'}_{loss}_"
            name = name + f"{'gnorm' if gradient_normalize else 'nognorm'}_"
            attack_args.update({
                'load_from': model,
                'attack_iterations': iterations,
                'attack_max_iterations': max_iterations,
                'attack_loss': loss,
                'attack_primal_optimizer': optimizer,
                'attack_dual_optimizer': df.attack_dual_optimizer,
                'attack_accelerated': accelerated,
                'attack_gradient_normalize': gradient_normalize,
                'working_dir': working_dir
            })
            if norm == 'li':
                use_sign = df.attack_use_sign
                attack_args.update({'attack_use_sign': use_sign})
                if not gradient_normalize and use_sign:
                    continue
                elif gradient_normalize and use_sign:
                    name = name[:-1] + "sign_"

            lr = df.attack_primal_lr
            min_lr = df.attack_primal_min_lr
            llr = df.attack_dual_lr
            C0 = df.attack_initial_const
            name = name + f"lr{lr:.2}_mlr{min_lr:.4}_llr{llr:.2}_C{C0}_"
            attack_args.update({
                'attack_primal_lr': lr,
                'attack_primal_min_lr': min_lr,
                'attack_dual_lr': llr,
                'attack_initial_const': C0
            })

            sampling_radius = df.attack_sampling_radius
            sampling_algorithm = df.attack_r0_init
            name = name + f"{sampling_algorithm}_R{sampling_radius}_"
            attack_args.update({
                'attack_r0_init': sampling_algorithm,
                'attack_sampling_radius': sampling_radius
            })

            lr_decay = df.attack_lr_decay
            finetune = df.attack_finetune
            use_proxy = df.attack_use_proxy_constraint
            name = name + f"{'decay' if lr_decay else 'nodecay'}_"
            name = name + f"{'finetune' if finetune else 'nofinetune'}_"
            name = name + f"{'proxy' if use_proxy else 'noproxy'}_"
            attack_args.update({
                'attack_lr_decay':
                lr_decay,
                'attack_finetune':
                finetune,
                'attack_use_proxy_constraint':
                use_proxy
            })

            p = [
                s.name[:-1] for s in list(Path(working_dir).glob("*"))
            ]
            if name in p:
                continue
            if norm != 'l2':
                print(
                    generate_test_optimizer_lp(
                        name=name,
                        ignored_flags=['attack_use_sign'],
                        **attack_args))


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
        flags.FLAGS._flags().clear()
        importlib.reload(test_optimizer_lp_madry)
        _, attack_klass = lp_attacks[attack]
        import_klass_kwargs_as_flags(attack_klass, 'attack_')
        test_lp_config(attack)
