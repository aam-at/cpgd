#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tikzplotlib

# set plot style
plt.style.use("seaborn-paper")
sns.set_style("whitegrid")
sns.set_style("ticks")

params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.rcParams.update(params)

# latex export
tikzplotlib.Flavors.latex.preamble()

class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


ROOT = "../results_mnist/"
results = DeepDict(DeepDict(DeepDict(DeepDict(list))))
shape = (501,)
for type, norm, attack in itertools.product(["plain", "linf", "l2"],
                                            ["l1", "l2", "li", "l0"],
                                            ["our_%(norm)s", "pgd"]):
    attack = attack % locals()
    base_dir = Path(ROOT) / f"test_{type}" / norm / attack
    load_dirs = list(base_dir.glob("*avg_acc*"))
    for load_dir in load_dirs:
        avg_acc = np.load(load_dir / "avg_acc.npy").mean(-1)
        assert avg_acc.shape == shape
        results[norm][type][attack][str(load_dir)[:-2]].append(avg_acc)

iterations = np.arange(0, 501)
for type, norm, attack in itertools.product(["plain", "linf", "l2"],
                                            ["l1", "l2", "li", "l0"],
                                            ["our_%(norm)s"]):
    attack = attack % locals()
    n = {"l1": 1, "l2": 2, "li": "\infty", "l0": 0}[norm]
    m = {"plain": "plain", "linf": "$l_{\infty}$-at", "l2": "$l_2$-at"}[type]

    # plot results
    fig, ax = plt.subplots()
    for load_dir, avg_acc in results[norm][type][attack].items():
        avg_acc = 100 * np.array(avg_acc)
        ci = 1.96 * np.std(avg_acc, 0)
        avg_acc = avg_acc.mean(0)

        fn = ax.plot
        line, = fn(iterations, avg_acc, linestyle="-", label=f"{load_dir}")
        ax.fill_between(iterations, (avg_acc - ci), (avg_acc + ci), alpha=0.1)
    # configure style of the chart
    size_inches = fig.get_size_inches()
    # s = size_inches[0] / 3.5
    # fig.set_size_inches(w=3.5, h=size_inches[1] / s)
    ax.set_xlim((0, 501))
    ax.set_xticks([1, 50] + list(range(100, 501, 100)))
    ax.grid(True)
    ax.spines["top"].set_color('.8')
    ax.spines["top"].set_linewidth(.8)
    ax.spines["right"].set_color('.8')
    ax.spines["right"].set_linewidth(.8)
    ax.tick_params(axis='x', direction="in", pad=2, labelsize=10, length=2)
    ax.tick_params(axis='y', direction="in", pad=1, labelsize=10, length=2)
    # plt.legend(loc="upper right")
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

    # tikzplotlib.clean_figure()
    # tikzplotlib.save(f"{norm}_{type}.tikz")
    plt.savefig(f"{norm}_{type}.pgf", bbox_inches="tight")

