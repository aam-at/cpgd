#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tikzplotlib
from matplotlib import rc
from matplotlib.ticker import NullFormatter, StrMethodFormatter
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes

# set plot style
plt.style.use("seaborn-paper")
sns.set_style("whitegrid")
sns.set_style("ticks")

params = {'legend.fontsize': 10, 'legend.handlelength': 2}
plt.rcParams.update(params)

## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# latex export
tikzplotlib.Flavors.latex.preamble()


class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


results = DeepDict(DeepDict(DeepDict(DeepDict(list))))
for type, norm, attack in itertools.product(["plain", "linf", "l2"], ["li", "l2", "l1", "l0"],
                                            ["our_%(norm)s", "pgd"]):
    attack = attack % locals()
    if attack == "pgd":
        ROOT = "../results_mnist/"
    else:
        ROOT = "../results_mnist_plot/"
    base_dir = Path(ROOT) / f"test_{type}" / norm / attack
    load_dirs = list(base_dir.glob("*avg_acc_*"))
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
    load_dirs = sorted(load_dirs, key=sort_fn)
    for load_dir in load_dirs:
        name = str(load_dir.name)[:-2]
        if attack.startswith("our"):
            if "init1.0_min0.01" not in name:
                continue
            # if "noema" in name:
            #     continue
            # if "_n100_" not in name and "_n250_" not in name:
            #     continue
            if "_n250_" not in name:
                continue
            # if "_n100_" in name and "C0.1" in name:
            #     continue
            if "R0.5" in name:
                continue
        if attack == "pgd":
            if "norand" not in name:
                continue
        avg_acc = np.load(load_dir / "avg_acc.npy")
        if avg_acc.shape[0] != 501:
            continue
        avg_acc = avg_acc.mean(-1)
        results[norm][type][attack][name].append(avg_acc)

# iterations = np.arange(0, 501)
for type, norm in itertools.product(["plain", "linf", "l2"],
                                    ["li", "l2", "l1", "l0"]):
    n = {"l1": 1, "l2": 2, "li": "\infty", "l0": 0}[norm]
    mo = {"plain": "plain", "linf": "$l_{\infty}$-at", "l2": "$l_2$-at"}[type]

    # plot results
    fig, ax = plt.subplots()
    with_axins = False
    if norm == "li":
        if type != "linf":
            with_axins = True
    elif norm == "l2":
        if type == "plain":
            with_axins = True
    elif norm == "l0":
        if type != "linf":
            with_axins = True
    if with_axins:
        axins = zoomed_inset_axes(ax, 5, loc=7) # zoom = 6
    d = {}
    m = 10000000
    for attack in ["our_%(norm)s", "pgd"]:
        attack = attack % locals()
        for load_dir, acc in results[norm][type][attack].items():
            acc = 100 * np.array(acc)[:, -1].mean()
            d[load_dir] = acc
            print(load_dir, acc)
            if acc < m:
                m = acc


        for load_dir, avg_acc in results[norm][type][attack].items():
            avg_acc = 100 * np.array(avg_acc)
            ci = 1.96 * np.std(avg_acc, 0)
            avg_acc = avg_acc.mean(0)

            fn = ax.semilogy
            iterations = np.arange(avg_acc.shape[0])
            if "pgd" in load_dir:
                label = f"PGD-$l_{n}$ on {mo}"
                color = "#67a9cf"
            else:
                if "C0.1" in load_dir:
                    label = f"Our-$l_{n}$ with C=0.1"
                    color = "#d7191c"
                elif "C10" in load_dir:
                    label = f"Our-$l_{n}$ with C=10"
                    color = "#ef8a62"
                elif "C1" in load_dir:
                    label = f"Our-$l_{n}$ with C=1"
                    color = "#1a9641"
            line, = fn(iterations, avg_acc, linestyle="-", label=label, color=color)
            ax.fill_between(iterations, (avg_acc - ci), (avg_acc + ci),
                            alpha=0.1)
            if with_axins:
                axins.plot(iterations[-21:], avg_acc[-21:], linestyle="-", label=label, color=color)

    if with_axins:
        axins.tick_params(axis='x', direction="in", pad=1, labelsize=8, length=2)
        axins.tick_params(axis='y', direction="in", pad=1, labelsize=8, length=2)
        mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    # configure style of the chart
    size_inches = fig.get_size_inches()
    s = size_inches[0] / 3.5
    fig.set_size_inches(w=3.5, h=size_inches[1] / s)
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ymin, ymax = ax.get_ylim()
    ax.set_xlim((0, 501))
    ax.set_xticks([1] + list(range(100, 501, 100)))
    a = list(np.linspace((ymin // 10 + 1) * 10, 100, int(10 - ymin // 10))) + [ymin//10 * 10 + 5]
    ax.set_yticks(a)
    ax.grid(True)
    ax.spines["top"].set_color('.8')
    ax.spines["top"].set_linewidth(.8)
    ax.spines["right"].set_color('.8')
    ax.spines["right"].set_linewidth(.8)
    ax.tick_params(axis='x', direction="in", pad=2, labelsize=9, length=2)
    ax.tick_params(axis='y', direction="in", pad=1, labelsize=9, length=2)
    handles, labels = ax.get_legend_handles_labels()
    def sort_fn(n):
        if "PGD" in n:
            return 1
        else:
            return 10
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: sort_fn(x[0])))
    if type != "linf" or norm == "li":
        ax.legend(handles, labels, bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
    else:
        ax.legend(handles, labels)

    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"{norm}_{type}.tikz")
    plt.savefig(f"avg_acc_{norm}_{type}.pdf", bbox_inches="tight")
