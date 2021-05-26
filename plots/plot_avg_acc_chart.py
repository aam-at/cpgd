#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

# set plot style
plt.style.use("seaborn-paper")
sns.set_style("whitegrid")
sns.set_style("ticks")

params = {'legend.fontsize': 8, 'legend.handlelength': 2}
plt.rcParams.update(params)

## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


def sort_fn(name):
    name = str(name)
    if "pgd" in name:
        return 100000
    elif "C0.1" in name:
        return 10
    elif "C10" in name:
        return 1000
    else:
        return 100


def group_fn(name):
    name = str(name)[:-2]
    if "pgd" in name:
        return "pgd"
    name = "_".join(name.split("_")[:-1])
    return name


for dataset in ["mnist", "cifar10"]:
    i_ = []
    for type, norm in itertools.product(
        ["plain", "linf", "l2"],
        ["li", "l2", "l1", "l0"],
    ):
        n = {"l1": 1, "l2": 2, "li": "\infty", "l0": 0}[norm]
        m = {
            "plain": "plain",
            "linf": "$l_{\infty}$-at",
            "l2": "$l_2$-at"
        }[type]

        results = DeepDict(DeepDict(list))
        for attack in ["our_%(norm)s", "pgd"]:
            attack = attack % locals()
            ROOT = f"../results_{dataset}_plot/"
            base_dir = Path(ROOT) / f"test_{type}" / norm / attack
            load_dirs = list(base_dir.glob("*avg_acc_*"))
            dirs = []
            for load_dir in load_dirs:
                name = str(load_dir.name)[:-2]
                if attack.startswith("our"):
                    # if "exp_init1.0_min0.01_flr_exp_init0.01_min0.0001" not in name:
                    #     continue
                    if "R0.0" not in name:
                        continue
                if attack == "pgd":
                    if "norand" not in load_dir.name:
                        continue
                dirs.append(load_dir)
            dirs = sorted(dirs, key=group_fn)
            groups = itertools.groupby(dirs, key=group_fn)
            best_group_key = None
            best_group_acc = {}
            best_group_value = 1e6
            for gr_key, gr_value in groups:
                accs = {}
                for load_dir in sorted(gr_value, key=sort_fn):
                    name = str(load_dir)[:-2]
                    avg_acc = 100 * np.load(load_dir / "avg_acc.npy")
                    avg_acc = avg_acc.mean(-1)
                    if attack == "pgd":
                        acc0 = avg_acc[0, 0]
                        iters = 100
                        mul = 5
                        for i in range(10):
                            iters = int(500 / mul)
                            mul = acc0
                            for i in range(5):
                                mul += avg_acc[i, iters]
                            mul /= 100
                        iters = math.ceil(iters * acc0 / 100)
                        i_.append(iters)
                        avg_acc = avg_acc[:, :iters + 1]
                        acc = [np.array([acc0] * 5).reshape(5, 1)]
                        for i in range(5):
                            acc_i = []
                            acc2 = []
                            for j in range(i):
                                acc_i.append(np.array([avg_acc[j, -1]] *
                                                      iters))
                            if i > 0:
                                m2 = np.array(acc_i).min()
                            else:
                                m2 = 1000
                            acc_i.extend([np.minimum(avg_acc[i, 1:], m2)] *
                                         (5 - i))
                            acc_i = np.array(acc_i)
                            acc.append(acc_i)
                        avg_acc = np.hstack(acc).mean(0)
                        total_iters = avg_acc.shape[0]
                        xs = np.arange(total_iters) / total_iters
                        xs_new = np.arange(500 + 1) / 500
                        avg_acc = np.interp(xs_new, xs, avg_acc)
                    accs[name] = avg_acc
                if attack != "pgd" and len(accs) != 3:
                    continue
                mean_acc = np.array(list(accs.values()))[:, 500].mean()
                if mean_acc < best_group_value:
                    best_group_key = gr_key
                    best_group_acc = accs
                    best_group_value = mean_acc
            results[attack].update(best_group_acc)

        # plot results
        fig, ax = plt.subplots()
        fn = ax.semilogy
        if dataset == "cifar10":
            fixins = {
                'plain': {
                    'li': True,
                    'l2': True,
                    'l1': False,
                    'l0': False
                },
                'linf': {
                    'li': True,
                    'l2': False,
                    'l1': False,
                    'l0': False
                },
                'l2': {
                    'li': True,
                    'l2': True,
                    'l1': False,
                    'l0': False
                }
            }
        else:
            fixins = {
                'plain': {
                    'li': True,
                    'l2': True,
                    'l1': False,
                    'l0': True
                },
                'linf': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': False
                },
                'l2': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': True
                }
            }
        with_axins = fixins[type][norm]
        if with_axins:
            if dataset == "cifar10":
                axins = zoomed_inset_axes(ax,
                                          8,
                                          loc=7,
                                          bbox_to_anchor=(224.0,
                                                          82.0))  # zoom = 6
            else:
                axins = zoomed_inset_axes(ax,
                                          8,
                                          loc=7,
                                          bbox_to_anchor=(224.0,
                                                          82.0))  # zoom = 6
        for attack in ["our_%(norm)s", "pgd"]:
            attack = attack % locals()
            for load_dir, avg_acc in results[attack].items():
                iterations = np.arange(avg_acc.shape[0])
                if "pgd" in load_dir:
                    label = f"PGD-$l_{n}$"
                    color = "#67a9cf"
                else:
                    if "C0.1" in load_dir:
                        label = f"Our-$l_{n}$ with C=0.1"
                        color = "#ef8a62"
                    elif "C10" in load_dir:
                        label = f"Our-$l_{n}$ with C=10"
                        color = "#d6604d"
                    elif "C1" in load_dir:
                        label = f"Our-$l_{n}$ with C=1"
                        color = "#5aae61"
                line, = fn(iterations,
                           avg_acc,
                           linestyle="-",
                           linewidth=1,
                           label=label,
                           color=color)
                if with_axins:
                    axins.plot(iterations[480:501],
                               avg_acc[480:501],
                               linestyle="-",
                               label=label,
                               color=color)

        if with_axins:
            axins.tick_params(axis='x',
                              direction="in",
                              pad=1,
                              labelsize=7,
                              length=2)
            axins.tick_params(axis='y',
                              direction="in",
                              pad=1,
                              labelsize=7,
                              length=2)
            mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
        # configure style of the chart
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ymin, ymax = ax.get_ylim()
        ax.set_xlim((0, 501))
        ax.set_xticks([1, 50] + list(range(100, 501, 100)))
        a = list(np.linspace((ymin // 10 + 1) * 10, 100,
                             int(10 - ymin // 10))) + [ymin // 10 * 10 + 5]
        ax.set_yticks(a)
        ax.grid(True)
        ax.spines["top"].set_color('.8')
        ax.spines["top"].set_linewidth(.8)
        ax.spines["right"].set_color('.8')
        ax.spines["right"].set_linewidth(.8)
        ax.tick_params(axis='x', direction="in", pad=2, labelsize=8, length=2)
        ax.tick_params(axis='y', direction="in", pad=1, labelsize=8, length=2)
        handles, labels = ax.get_legend_handles_labels()

        def sort_fn2(n):
            if "PGD" in n:
                return 1
            else:
                return 10

        labels, handles = zip(
            *sorted(zip(labels, handles), key=lambda x: sort_fn2(x[0])))
        if dataset == "cifar10":
            legend_auto = {
                'plain': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': False
                },
                'linf': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': False
                },
                'l2': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': False
                }
            }
        else:
            legend_auto = {
                'plain': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': False
                },
                'linf': {
                    'li': True,
                    'l2': True,
                    'l1': True,
                    'l0': True
                },
                'l2': {
                    'li': False,
                    'l2': False,
                    'l1': False,
                    'l0': False
                }
            }
        if legend_auto[type][norm]:
            ax.legend(handles, labels)
        else:
            ax.legend(handles,
                      labels,
                      bbox_to_anchor=(1, 1),
                      loc=1,
                      borderaxespad=0)

        size_inches = fig.get_size_inches()
        s = size_inches[0] / 3.5
        fig.set_size_inches(w=3.5, h=size_inches[1] / s)
        plt.savefig(f"{dataset}_avg_acc_{norm}_{type}.pdf",
                    bbox_inches="tight")
    print(dataset, np.mean(i_))
