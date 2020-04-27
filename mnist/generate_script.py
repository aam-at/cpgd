from __future__ import absolute_import, division, print_function

import importlib
import itertools
import subprocess
from pathlib import Path
import importlib

import numpy as np

from lib.generate_script import generate_test_optimizer
from lib.parse_logs import parse_test_optimizer_log

models = [
    './models/mnist_weights_plain.mat', './models/mnist_weights_linf.mat',
    './models/mnist_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')


def generate_test_optimizer_lp(norm, load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', norm, load_from,
                                   **kwargs)


def generate_test_bethge_lp(norm, load_from, **kwargs):
    return generate_test_optimizer('test_bethge_attack', norm, load_from,
                                   **kwargs)


def test_lp_config(norm, runs=1, master_seed=1):
    assert norm in ['l0', 'li', 'l1', 'l2']
    num_images = {'l0': 1000, 'li': 1000, 'l1': 1000, 'l2': 500}[norm]
    batch_size = 500
    attack_args = {
        'norm': norm,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size
    }
    name = "mnist_lin_lambd_c2_"
    for (model, iterations, max_iterations, loss, optimizer, accelerated,
         gradient_normalize,
         use_sign,
         adaptive_momentum,
         dual_ema) in itertools.product(models, [500], [1], ["cw"],
                                        ["sgd"], [False, True],
                                        [False, True],
                                        [False, True],
                                        [False],
                                        [True, False]):
        max_iterations *= iterations
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist_10/test_{norm}_{type}"
        name0 = name + f"{norm}_{type}_{iterations}_{max_iterations // 1000}k_"
        name0 = name0 + f"{optimizer if not accelerated else optimizer + '_apg'}_{loss}_"
        name0 = name0 + f"{'gnorm' if gradient_normalize else 'nognorm'}_"
        attack_args0 = attack_args.copy()
        if norm == 'li':
            attack_args0.update({'attack_use_sign': use_sign})
            if not gradient_normalize:
                continue
            elif gradient_normalize and use_sign:
                name0 = name0[:-1] + "sign_"
            else:
                continue
        elif use_sign:
            continue
        if not accelerated and adaptive_momentum:
            continue
        if gradient_normalize and norm != 'li':
            continue
        name0 = name0 + f"{'adaptive' if adaptive_momentum else 'nonadaptive'}_"
        name0 = name0 + f"{'ema' if dual_ema else 'noema'}_"
        attack_args0.update({
            'load_from': model,
            'attack_iterations': iterations,
            'attack_max_iterations': max_iterations,
            'attack_loss': loss,
            'attack_primal_optimizer': optimizer,
            'attack_gradient_normalize': gradient_normalize,
            'attack_accelerated': accelerated,
            'attack_adaptive_momentum': adaptive_momentum,
            'attack_dual_optimizer': 'sgd',
            'attack_dual_ema': dual_ema,
            'working_dir': working_dir
        })
        if type == 'plain':
            if norm == 'l0':
                lr_space = [0.1, 0.5]
            else:
                lr_space = [0.1, 0.5]
        elif type == 'l2':
            if norm == 'li':
                lr_space = [0.1, 0.5]
            else:
                lr_space = [1.0, 0.5]
        else:
            if norm == 'l2':
                lr_space = [0.1, 0.5]
            else:
                lr_space = [0.1, 0.5]
        lr_space = [0.01, 0.05] + list(np.linspace(0.1, 0.5, 5)) + [1.0]
        for lr, min_lr, llr, C0 in itertools.product(lr_space, [0.1, 0.01, 0.001], [0.1], [1., 0.1, 0.01]):
            if lr < min_lr:
                continue
            attack_args1 = attack_args0.copy()
            attack_args1.update({
                'attack_primal_lr': lr,
                'attack_primal_min_lr': min_lr,
                'attack_dual_lr': llr,
                'attack_initial_const': C0
            })
            name1 = name0 + f"lr{lr:.2}_mlr{min_lr:.4}_llr{llr:.2}_C{C0:.4}_"
            for (sampling_radius,
                 sampling_algorithm) in itertools.product([0.5], ["uniform"]):
                attack_args2 = attack_args1.copy()
                attack_args2.update({
                    'attack_r0_init': sampling_algorithm,
                    'attack_sampling_radius': sampling_radius,
                })
                name2 = name1 + f"{sampling_algorithm}_R{sampling_radius}_"
                name2 = name2 + "initial_"
                for lr_decay, finetune, use_proxy in itertools.product(
                    [True, False], [True], [False]):
                    if lr == min_lr and lr_decay:
                        continue
                    attack_args3 = attack_args2.copy()
                    attack_args3.update({
                        'attack_lr_decay':
                        lr_decay,
                        'attack_finetune':
                        finetune,
                        'attack_use_proxy_constraint':
                        use_proxy
                    })
                    name3 = name2 + f"{'decay' if lr_decay else 'nodecay'}_"
                    name3 = name3 + f"{'finetune' if finetune else 'nofinetune'}_"
                    name3 = name3 + f"{'proxy' if use_proxy else 'noproxy'}_"
                    np.random.seed(master_seed)
                    p = [
                        s.name[:-1] for s in list(Path(working_dir).glob("*"))
                    ]
                    if name3 in p:
                        continue
                    for i in range(runs):
                        seed = np.random.randint(1000)
                        if True:
                            print(
                                generate_test_optimizer_lp(
                                    name=name3,
                                    seed=seed,
                                    ignored_flags=['attack_use_sign'],
                                    **attack_args3))


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
            'norm': norm,
            'load_from': model,
            'working_dir': working_dir
        })
        print(generate_test_bethge_lp(name=name, **attack_args0))


if __name__ == '__main__':
    for norm in ['l0', 'l1', 'l2', 'li']:
        if norm == 'l2':
            # test_lp_custom_config(norm)
            test_lp_config(norm)
