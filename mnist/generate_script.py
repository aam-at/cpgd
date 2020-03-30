from __future__ import absolute_import, division, print_function

import itertools
import subprocess
from pathlib import Path

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


def test_lp_config(norm, runs=1, master_seed=1):
    assert norm in ['l0', 'li', 'l1', 'l2']
    num_batches = {'l0': 10, 'li': 10, 'l1': 10, 'l2': 5}[norm]
    attack_args = {'norm': norm, 'num_batches': num_batches}
    name = "mnist_"
    for (model, iterations, max_iterations, loss, optimizer, accelerated,
         gradient_normalize,
         use_sign) in itertools.product(models, [100], [1000], ["cw"],
                                        ["adam"], [True, False], [True, False],
                                        [True, False]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist/test_{norm}_{type}"
        name0 = name + f"{norm}_{type}_{iterations}_{max_iterations // 1000}k_"
        name0 = name0 + f"{optimizer if not accelerated else 'apg'}_{loss}_"
        name0 = name0 + f"{'gnorm' if gradient_normalize else 'nognorm'}_"
        attack_args0 = attack_args.copy()
        attack_args0.update({
            'load_from': model,
            'attack_iterations': iterations,
            'attack_max_iterations': max_iterations,
            'attack_loss': loss,
            'attack_optimizer': optimizer,
            'attack_accelerated': accelerated,
            'attack_gradient_normalize': gradient_normalize,
            'working_dir': working_dir
        })
        if norm == 'li':
            attack_args0.update({'attack_use_sign': use_sign})
            if not gradient_normalize and use_sign:
                continue
            elif gradient_normalize and use_sign:
                name0 = name0[:-1] + "sign_"
        elif use_sign:
            continue
        for lr, llr, C0 in itertools.product(np.linspace(0.05, 0.5, 10),
                                             [0.1, 0.05, 0.01], [0.1, 0.5]):
            attack_args1 = attack_args0.copy()
            attack_args1.update({
                'attack_primal_lr': lr,
                'attack_primal_min_lr': lr / 10.0,
                'attack_dual_lr': llr,
                'attack_initial_const': C0
            })
            name1 = name0 + f"lr{lr:.2}_llr{llr:.2}_C{C0}_"
            for sampling_radius, sampling_algorithm in itertools.product(
                [0.5], ["uniform"]):
                attack_args2 = attack_args1.copy()
                attack_args2.update({
                    'attack_r0_init': sampling_algorithm,
                    'attack_sampling_radius': sampling_radius
                })
                name2 = name1 + f"{sampling_algorithm}_R{sampling_radius}_"
                for lr_decay, finetune, use_proxy in itertools.product(
                    [True, False], [True, False], [True, False]):
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
                        print(
                            generate_test_optimizer_lp(
                                name=name3,
                                seed=seed,
                                ignored_flags=['attack_use_sign'],
                                **attack_args3))

    df = parse_test_optimizer_log(
        "../results/mnist_final/test_l2_plain",
        export_test_params=[
            "attack_%s" % n for n in [
                "optimizer", "finetune", "primal_lr", "dual_lr", "r0_init",
                "sampling_radius", "initial_const", "tol", "max_iter"
            ]
        ])
    df = df[df["acc_l2_1.00"] <= 0.464]
    df = df[df["attack_max_iter"] == 1000]
    df = df.sort_values('l2', ascending=True)
    for model, attack_max_iter in itertools.product(models, [100000]):
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results/mnist_final/test_l2_{type}"

        for i, df_row in df.iterrows():
            opt = df_row.attack_optimizer
            lr = df_row.attack_primal_lr
            lr = 0.05
            llr = df_row.attack_dual_lr
            C0 = df_row.attack_initial_const
            r0_init = df_row.attack_r0_init
            sampling_radius = int(df_row.attack_sampling_radius)
            tol = df_row.attack_tol
            finetune = df_row.attack_finetune
            finetunestr = "finetune" if finetune else "nofinetune"

            np.random.seed(master_seed)
            name = f"mnist_l2_{type}_{attack_max_iter // 1000}k_{opt}_lr{lr}_llr{llr}_C{C0}_tol{tol}_r{r0_init}_R{sampling_radius}_{finetunestr}_{hostname}_"
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p:
                continue
            for i in range(runs):
                seed = np.random.randint(1e6)
                print(
                    generate_test_optimizer_l2(
                        load_from=model,
                        attack_finetune=finetune,
                        attack_optimizer=opt,
                        attack_max_iter=attack_max_iter,
                        attack_min_iter_per_start=0,
                        attack_max_iter_per_start=100,
                        attack_r0_init=r0_init,
                        attack_sampling_radius=sampling_radius,
                        attack_primal_lr=lr,
                        attack_dual_lr=llr,
                        attack_tol=tol,
                        attack_initial_const=C0,
                        num_batches=5,
                        working_dir=working_dir,
                        name=name,
                        seed=seed))


def test_li_config(runs=1, master_seed=1):
    for (model, max_iter, opt, tol, sampling_radius) in itertools.product(
            models, [1000], ["adam"], [0.005, 0.001, 0.0005],
        [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]):
        type = Path(model).stem.split("_")[-1]
        if type not in ['linf']:
            continue
        working_dir = f"../results/mnist_temp/test_li_{type}"
        attack_max_iter = max_iter
        finetune = True
        finetunestr = "finetune" if finetune else "nofinetune"
        for lr, llr, r0_init, C0 in itertools.product([0.05], [0.1],
                                                      ["sign", "uniform"],
                                                      [0.1, 0.5, 0.9]):
            np.random.seed(master_seed)
            name = f"mnist_li_{type}_{attack_max_iter//1000}k_{opt}_lr{lr}_llr{llr}_C{C0}_tol{tol}_r{r0_init}_R{sampling_radius}_{finetunestr}_{hostname}_"
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p:
                continue
            for i in range(runs):
                seed = np.random.randint(1e6)
                print(
                    generate_test_optimizer_li(
                        load_from=model,
                        attack_finetune=finetune,
                        attack_optimizer=opt,
                        attack_max_iter=attack_max_iter,
                        attack_min_iter_per_start=0,
                        attack_max_iter_per_start=100,
                        attack_r0_init=r0_init,
                        attack_sampling_radius=sampling_radius,
                        attack_primal_lr=lr,
                        attack_dual_lr=llr,
                        attack_tol=tol,
                        attack_initial_const=C0,
                        num_batches=10,
                        working_dir=working_dir,
                        name=name,
                        seed=seed))


if __name__ == '__main__':
    for norm in ['l0', 'l1', 'l2', 'li']:
        test_lp_config(norm)
