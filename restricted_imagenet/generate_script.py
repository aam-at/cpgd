"""This script can be used to generate test configs for the compared attacks.
"""
from __future__ import absolute_import, division, print_function

import ast
import importlib
import itertools
import subprocess
from pathlib import Path

import numpy as np
from absl import flags
from lib.attack_lp import ProximalPrimalDualGradientAttack
from lib.generate_script import (cleanflags, count_number_of_lines,
                                 format_name, generate_test_optimizer)
from lib.parse_logs import parse_log
from lib.tf_utils import ConstantDecay, ExpDecay, LinearDecay
from lib.utils import (format_float, import_func_annotations_as_flags,
                       import_klass_annotations_as_flags)

from config import test_model_thresholds

models = {
    'plain': './models/train_224_nat_slim',
    'linf': './models/train_224_robust_eps_0.005_lp_inf_slim',
    'l2': './models/train_224_robust_eps_1.0_lp_2_slim'
}
hostname = subprocess.getoutput('hostname')
basedir = "results_imagenet"

NUM_IMAGES = 1000

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


def load_logs(load_dir):
    logs = []
    for dir in glob.glob(load_dir + "*"):
        log = parse_log(dir, export_test_params=True)
        logs.append(log)
    df = pd.concat(logs, ignore_index=True)
    return df


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
        'attack_simultaneous_updates': [True, False],
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

    num_images = {'l0': 500, 'li': 500, 'l1': 500, 'l2': 500}[norm]
    batch_size = 50
    attack_args = {
        'attack': attack,
        'num_batches': num_images // batch_size,
        'batch_size': batch_size,
        'seed': 1
    }

    existing_names = []
    for type in models.keys():
        working_dir = f"../results/imagenet_10/test_{type}_{norm}"
        attack_args.update({'load_from': models[type], 'working_dir': working_dir})

        # parse test log
        df = parse_test_log(Path(working_dir) / f"imagenet_{type}_{attack}_*",
                            export_test_params=test_params)
        df = df[df.attack == attack]
        df = df.sort_values(norm)
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
            if attack_args['attack_loop_r0_sampling_epsilon'] != 0.1:
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
            attack_args['working_dir'] = f"../results/imagenet_final/test_{type}_{norm}"

            # change args
            j += 1
            for upd, r, R in itertools.product([True, False], [0.1], [1, 10]):
                attack_args['attack_simultaneous_updates'] = upd
                attack_args['attack_loop_number_restarts'] = R
                attack_args['attack_loop_r0_sampling_epsilon'] = r
                # generate unique name
                base_name = f"imagenet_{type}"
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


@count_number_of_lines
@cleanflags
def pgd_config(norm, seed=123):
    import test_pgd
    from test_pgd import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_pgd)
    import_flags(norm)

    num_images = 1000
    batch_size = 250
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

    for type in models.keys():
        working_dir = f"../{basedir}/test_{type}/{norm}/pgd"
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        for attack_arg_value in itertools.product(*attack_grid_args.values()):
            attack_args = dict(zip(attack_arg_names, attack_arg_value))
            attack_args.update({
                'load_from': models[type],
                'working_dir': working_dir,
            })
            for eps, eps_scale in itertools.product(
                    test_model_thresholds[type][norm],
                [1, 2, 5, 10, 25, 50, 100]):
                attack_args.update({
                    'attack_eps': eps,
                    'attack_eps_iter': round(eps / eps_scale, 6)
                })
                name = f"""imagenet_pgd_{type}_{norm}_{attack_args['attack_loss']}_
n{attack_args['attack_nb_iter']}_N{attack_args['attack_nb_restarts']}_
eps{format_float(eps, 4)}_epss{eps_scale}_""".replace("\n", "")
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
                        name = f"""imagenet_pgd_{type}_{norm}_{attack_args['attack_loss']}_
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


def daa_config(seed=123):
    import test_daa
    from test_daa import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_daa)
    import_flags("blob")

    batch_size = 50
    norm = 'li'
    attack_args = {
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for type in models.keys():
        for nb_iter, nb_restarts, method, eps, eps_scale in itertools.product(
                [200], [1, 5], ['dgf', 'blob'], test_model_thresholds[type][norm], [1, 2, 5, 10, 25, 50, 100]):
            working_dir = f"../results_imagenet/test_{type}/{norm}/daa"
            attack_args.update({
                'load_from': models[type],
                'working_dir': working_dir,
                'method': method,
                'attack_nb_restarts': nb_restarts,
                'attack_nb_iter': nb_iter,
                'attack_eps': eps,
                'attack_eps_iter': eps / eps_scale
            })
            name = f"imagenet_daa_{method}_{type}_n{nb_iter}_N{nb_restarts}_eps{eps}_epss{eps_scale}_"
            attack_args['name'] = name
            p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
            if name in p or name in existing_names:
                continue
            existing_names.append(name)
            print(generate_test_optimizer('test_daa', **attack_args))


# fab attacks
def fab_config(norm, seed=123):
    import test_fab
    from lib.fab import FABAttack

    flags.FLAGS._flags().clear()
    importlib.reload(test_fab)
    import_klass_annotations_as_flags(FABAttack, 'attack_')

    batch_size = 50
    attack_args = {
        'attack_norm': norm,
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for type, n_restarts in itertools.product(models.keys(), [1, 10]):
        # default params for imagenet
        # see page 12: https://openreview.net/pdf?id=HJlzxgBtwH
        n_iter = 100
        alpha_max = 0.05
        eta = 1.3
        beta = 0.9
        eps = {
            'plain' : {'li' : 0.02, 'l2' : 5.0, 'l1' : 100.0},
            'linf'  : {'li' : 0.08, 'l2' : 5.0, 'l1' : 250.0},
            'l2'    : {'li' : 0.08, 'l2' : 5.0, 'l1' : 250.0}
        }

        # params
        working_dir = f"../results_imagenet/test_{type}/{norm}/fab"
        attack_args.update(
        {
            'attack_n_iter': n_iter,
            'attack_n_restarts': n_restarts,
            'attack_alpha_max': alpha_max,
            'attack_eta': eta,
            'attack_beta': beta,
            'attack_eps': eps[type][norm],
            'working_dir': working_dir,
            'load_from': models[type]
        })
        name = f"imagenet_fab_{type}_{norm}_n{n_iter}_N{n_restarts}_"
        attack_args["name"] = name
        p = [
            s.name[:-1] for s in list(Path(working_dir).glob("*"))
        ]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_fab', **attack_args))


# foolbox attacks
def foolbox_config(norm, attack, seed=123):
    import test_foolbox
    from test_foolbox import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_foolbox)
    import_flags(norm, attack)

    batch_size = 50
    attack_grid_args = {
        'num_batches': [NUM_IMAGES // batch_size],
        'batch_size': [batch_size],
        'load_from': models,
        'attack': [attack],
        'norm': [norm],
        'seed': [seed]
    }
    if attack == 'df':
        # default params
        attack_grid_args.update({
            'attack_steps': [50],
            'attack_overshoot': [0.02],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_os{attack_args['attack_overshoot']}_"
    elif attack == 'cw':
        # default params
        attack_grid_args.update({
            'attack_steps': [10000],
            'attack_stepsize': [0.01],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_abort_early': [False],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'newton':
        # default params
        attack_grid_args.update({
            'attack_steps': [1000],
            'attack_stepsize': [0.01],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_lr{attack_args['attack_stepsize']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_steps': [1000],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_regularization': [0.05],
            'attack_abort_early': [False],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_b{attack_args['attack_regularization']}_C{attack_args['attack_initial_const']}_"
    elif attack == 'ddn':
        # default params for imagenet
        # see: http://openaccess.thecvf.com/content_CVPR_2019/papers/Rony_Decoupling_Direction_and_Norm_for_Efficient_Gradient-Based_L2_Adversarial_Attacks_CVPR_2019_paper.pdf
        attack_grid_args.update({
            'attack_steps': [1000],
            'attack_init_epsilon': [1.0],
            'attack_gamma': [0.05],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_foolbox_n{attack_args['attack_steps']}_eps{attack_args['attack_init_epsilon']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_imagenet/test_{type}/{norm}/{attack}"
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


def bethge_config(norm, seed=123):
    import test_bethge
    from test_bethge import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_bethge)
    import_flags(norm)

    batch_size = 50
    attack_args = {
        'norm': norm,
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for type, lr, num_decay in itertools.product(models.keys(), [1.0], [20]):
        working_dir = f"../results_imagenet/test_{type}/{norm}/bethge"
        attack_args.update({
            'load_from': models[type],
            'working_dir': working_dir,
            'attack_lr': lr,
            'attack_lr_num_decay': num_decay
        })
        name = f"imagenet_bethge_{type}_{norm}_lr{lr}_nd{num_decay}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_bethge', **attack_args))


def deepfool_config(norm, seed=123):
    import test_deepfool

    flags.FLAGS._flags().clear()
    importlib.reload(test_deepfool)

    assert norm in ['l2', 'li']
    batch_size = 50
    attack_args = {
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'norm': norm,
        'seed': seed
    }

    existing_names = []
    for type in models.keys():
        working_dir = f"../results_imagenet/test_{type}/{norm}/df"
        attack_args.update({
            'load_from': models[type],
            'working_dir': working_dir,
            'attack_overshoot': 0.02,
            'attack_max_iter': 50,
        })
        name = f"imagenet_deepfool_{type}_{norm}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_deepfool', **attack_args))


def sparsefool_config(seed=123):
    import test_sparsefool

    flags.FLAGS._flags().clear()
    importlib.reload(test_sparsefool)

    norm = 'l1'
    batch_size = 50
    attack_args = {
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for type, lambda_ in itertools.product(models.keys(), [1.0, 2.0, 3.0]):
        working_dir = f"../results_imagenet/test_{type}/{norm}/sparsefool"
        attack_args.update({
            'load_from': models[type],
            'working_dir': working_dir,
            'attack_epsilon': 0.02,
            'attack_max_iter': 20,
            'attack_lambda_': lambda_,
        })
        name = f"imagenet_sparsefool_{type}_{norm}_l{lambda_}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_sparsefool', **attack_args))


# ibm art attacks
def art_config(norm, attack, seed=123):
    import test_art
    from test_foolbox import import_flags

    flags.FLAGS._flags().clear()
    importlib.reload(test_art)
    import_flags(norm, attack)

    batch_size = 50
    attack_grid_args = {
        'num_batches': [NUM_IMAGES // batch_size],
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
            'attack_max_iter': [100, 1000],
            'attack_nb_grads': [10],
            'attack_epsilon': [0.02],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_art_n{attack_args['attack_max_iter']}_os{attack_args['attack_epsilon']}_"
    elif attack == 'cw':
        # default params
        if norm == "l2":
            attack_grid_args.update({
                'attack_max_iter': [10000],
                'attack_initial_const': [0.01],
                'attack_binary_search_steps': [9],
            })
            name_fn = lambda: f"imagenet_{type}_{attack}_art_n{attack_args['attack_max_iter']}_C{attack_args['attack_initial_const']}_"
        else:
            attack_grid_args.update({
                'attack_max_iter': [10000],
                'attack_eps': [0.3],
            })
            name_fn = lambda: f"imagenet_{type}_{attack}_art_n{attack_args['attack_max_iter']}_eps{attack_args['attack_eps']}_"
    elif attack == 'ead':
        # default params
        attack_grid_args.update({
            'attack_max_iter': [1000],
            'attack_initial_const': [0.01],
            'attack_binary_search_steps': [9],
            'attack_decision_rule': ['L1'],
            'attack_beta': [0.05],
        })
        name_fn = lambda: f"imagenet_{type}_{attack}_art_n{attack_args['attack_max_iter']}_b{attack_args['attack_beta']}_C{attack_args['attack_initial_const']}_"

    attack_arg_names = list(attack_grid_args.keys())
    existing_names = []

    for attack_arg_value in itertools.product(*attack_grid_args.values()):
        model = attack_arg_value[attack_arg_names.index('load_from')]
        type = Path(model).stem.split("_")[-1]
        working_dir = f"../results_imagenet/test_{type}/{norm}/{attack}"
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


def jsma_config(seed=123):
    batch_size = 50
    attack_args = {
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }

    existing_names = []
    for type, targets, theta, lib in itertools.product(
            models.keys(), ["all", "random", "second"],
            [1.0], ["cleverhans", "art"]):
        working_dir = f"../results_imagenet/test_{type}/{norm}/jsma"
        attack_args.update({
            'load_from': models[type],
            'working_dir': working_dir,
            'attack_targets': targets,
            'attack_theta': theta,
            'attack_gamma': 1.0,
            'attack_impl': lib
        })
        name = f"imagenet_jsma_{type}_{targets}_t{theta}_g1.0_lib{lib}_"
        attack_args['name'] = name
        p = [s.name[:-1] for s in list(Path(working_dir).glob("*"))]
        if name in p or name in existing_names:
            continue
        existing_names.append(name)
        print(generate_test_optimizer('test_jsma', **attack_args))


def one_pixel_attack_config(seed=123):
    batch_size = 50
    attack_args = {
        'num_batches': NUM_IMAGES // batch_size,
        'batch_size': batch_size,
        'seed': seed
    }
    norm = "l0"

    existing_names = []
    for type, iters, es in itertools.product(models.keys(), [100], [1]):
        working_dir = f"../results_imagenet/test_{type}/{norm}/one_pixel"
        attack_args.update({
            'load_from': models[type],
            'working_dir': working_dir,
            'attack_iters': iters,
            'attack_es': es,
        })
        for threshold in test_model_thresholds[type]["l0"]:
            attack_args['attack_threshold'] = threshold
            name = f"imagenet_one_pixel_{type}_es{es}_i{iters}_t{threshold}_"
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
