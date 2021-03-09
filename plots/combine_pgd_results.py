#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np


class DeepDict(defaultdict):
    def __call__(self):
        return DeepDict(self.default_factory)


ROOT = "../results_mnist/"
results = DeepDict(DeepDict(DeepDict(DeepDict(list))))
for type, norm in itertools.product(["plain", "linf", "l2"],
                                    ["l0", "l1", "l2", "li"]):
    base_dir = Path(ROOT) / f"test_{type}" / norm / "pgd"
    load_dirs = list(base_dir.glob("*pgd_acc*"))
    for load_dir in sorted(list(load_dirs)):
        f = list(load_dir.glob("*.npy"))[0]
        p = "_".join(f.parent.name.split("_")[:6]).replace("acc", "avg_acc")
        if "norand" in str(load_dir):
            p = p + "_norand"
        results[norm][type][f.stem][p].append(np.load(f))
    avg_acc_th = DeepDict(OrderedDict)
    for n in sorted(results[norm][type].keys(), key=lambda x: float(x.replace("acc_", ""))):
        n_ = float(n.replace("acc_", ""))
        for k in results[norm][type][n].keys():
            k = k.replace("_cw", "").replace("_ce", "")
            avg_acc_th[k][n] = 0
    for n in results[norm][type]:
        x = DeepDict(defaultdict)
        for k, v in results[norm][type][n].items():
            k = k.replace("_cw", "").replace("_ce", "")
            x[k][n] = 1000
        for k, v in results[norm][type][n].items():
            avg_acc_at = np.array(v)
            avg_acc_at = avg_acc_at.cumprod(1)
            m = np.mean(avg_acc_at, (0, 2))[-1]
            k = k.replace("_cw", "").replace("_ce", "")
            if m < x[k][n]:
                x[k][n] = m
                avg_acc_th[k][n] = avg_acc_at
    avg_acc = DeepDict(list)
    for n, v in avg_acc_th.items():
        for k, v in v.items():
            avg_acc[n].append(v)
        try:
            assert np.array(avg_acc[n]).shape[0] == 5
            avg_acc[n] = np.array(avg_acc[n]).mean(0)
        except:
            print(n)
    for n in avg_acc.keys():
        for i, acc in enumerate(avg_acc[n]):
            name = n + f"_{i}"
            dir = load_dir.parent / name
            dir.mkdir()
            np.save(str(dir / "avg_acc.npy"), acc)
