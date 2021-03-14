#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# set plot style
plt.style.use("seaborn-paper")
sns.set_style("whitegrid")
sns.set_style("ticks")

params = {'legend.fontsize': 10, 'legend.handlelength': 2}
plt.rcParams.update(params)


class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


dataset = "mnist"
ROOT = f"../results_{dataset}_plot/"
results = DeepDict(DeepDict(DeepDict(int)))
shape = (100, 1000)
for type, norm, attack in itertools.product(["plain", "linf", "l2"],
                                            ["l1", "l2", "li"],
                                            ["our_%(norm)s", "fab"]):
    attack = attack % locals()
    base_dir = Path(ROOT) / f"test_{type}" / norm / attack
    load_dirs = list(base_dir.glob("*norms*"))
    for i, load_dir in enumerate(load_dirs):
        try:
            attack_norms = np.load(load_dir / "norms.npy")
            assert attack_norms.shape == shape
            attack_norms[attack_norms > 1e4] = np.nan
            is_corr = attack_norms[0] != 0
            attack_norms = attack_norms[:, is_corr]
            attack_norms = np.array([n[~np.isnan(n)].mean() for n in attack_norms])
            if attack in results[norm][type]:
                prev = results[norm][type][attack]
                if attack_norms[0] < prev[0]:
                    results[norm][type][attack] = attack_norms
            else:
                results[norm][type][attack] = attack_norms
        except:
            pass

restarts = np.arange(1, 101)
for norm in ["l1", "l2", "li"]:
    n = {"l1": 1, "l2": 2, "li": "\infty"}[norm]
    for type in ["plain", "linf", "l2"]:
        m = {
            "plain": "plain",
            "linf": "$l_{\infty}$-at",
            "l2": "$l_2$-at"
        }[type]
        fig, ax = plt.subplots()
        fn = ax.plot
        labels = {'fab': f"FAB-$l_{n}$",
                  'our': f"Our-$l_{n}$"}
        colors = {'fab': "#67a9cf",
                  "our": "#ef8a62"}
        for attack, attack_norms in results[norm][type].items():
            line, = fn(restarts,
                       attack_norms,
                       linestyle="-",
                       label=labels[attack.split("_")[0]],
                       color=colors[attack.split("_")[0]])
        # configure style of the chart
        size_inches = fig.get_size_inches()
        s = size_inches[0] / 3.5
        fig.set_size_inches(w=3.5, h=size_inches[1] / s)
        ax.set_xlim((0, 100))
        ax.set_xticks([1] + list(range(10, 101, 10)))
        ax.grid(True)
        ax.spines["top"].set_color('.8')
        ax.spines["top"].set_linewidth(.8)
        ax.spines["right"].set_color('.8')
        ax.spines["right"].set_linewidth(.8)
        ax.tick_params(axis='x', direction="in", pad=2, labelsize=9, length=2)
        ax.tick_params(axis='y', direction="in", pad=1, labelsize=9, length=2)
        # plt.legend(loc="upper right")
        plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)

        # tikzplotlib.clean_figure()
        # tikzplotlib.save(f"{norm}_{type}.tikz")
        plt.savefig(f"{dataset}_avg_norm_{norm}_{type}.pdf",
                    bbox_inches="tight")
