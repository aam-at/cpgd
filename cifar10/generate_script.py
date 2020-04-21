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
    './models/cifar10_weights_plain.mat', './models/cifar10_weights_linf.mat',
    './models/cifar10_weights_l2.mat'
]
hostname = subprocess.getoutput('hostname')


def generate_test_optimizer_lp(norm, load_from, **kwargs):
    return generate_test_optimizer('test_optimizer_lp_madry', norm, load_from,
                                   **kwargs)


def test_lp_config(norm, runs=1, master_seed=1):
    assert norm in ['l0', 'li', 'l1', 'l2']
    num_batches = {'l0': 10, 'li': 10, 'l1': 10, 'l2': 5}[norm]
    num_batches = 1
    attack_args = {'norm': norm, 'num_batches': num_batches}
    name = "cifar10_lin_"
    for (model, iterations, max_iterations, loss, optimizer, accelerated,
         gradient_normalize,
         use_sign,
         adaptive_momentum,
         dual_ema) in itertools.product(models, [500], [1], ["cw"],
                                        ["sgd"], [False, True],
                                        [False, True],
                                        [False, True],
                                        [True, False],
                                        [True]):
        max_iterations *= iterations
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/cifar10_hs/test_{norm}_{type}"
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
        for lr, min_lr, llr, C0 in itertools.product(lr_space, [0.01], [0.1], [0.01]):
            attack_args1 = attack_args0.copy()
            attack_args1.update({
                'attack_primal_lr': lr,
                'attack_primal_min_lr': min_lr,
                'attack_dual_lr': llr,
                'attack_initial_const': C0 / lr
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
                    [True, False], [True], [False, True]):
                    # if not lr_decay and norm != 'l0':
                    #     continue
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
                        if norm == 'l2':
                            print(
                                generate_test_optimizer_lp(
                                    name=name3,
                                    seed=seed,
                                    ignored_flags=['attack_use_sign'],
                                    **attack_args3))

if __name__ == '__main__':
    for norm in ['l0', 'l1', 'l2', 'li']:
        test_lp_config(norm)
