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
results = DeepDict(DeepDict(DeepDict(list)))
shape = (501, 1000)
for type, norm in itertools.product(["plain", "linf", "l2"],
                                    ["l0", "l1", "l2", "li"]):
    base_dir = Path(ROOT) / f"test_{type}" / norm / "pgd"
    load_dirs = list(base_dir.glob("*pgd_acc*"))
    for load_dir in load_dirs:
        f = list(load_dir.glob("*.npy"))[0]
        results[norm][type][f.stem].append(np.load(f))
    avg_acc_th = OrderedDict()
    for k in sorted([float(k.replace("acc_", "")) for k in results[norm][type].keys()]):
        avg_acc_th[k] = 0
    for k, v in results[norm][type].items():
        avg_acc_at = np.array(v)
        avg_acc_th[float(k.replace("acc_", ""))] = avg_acc_at
    avg_acc = []
    for k, v in avg_acc_th.items():
        avg_acc.append(v)
    avg_acc = np.array(avg_acc).mean(0)
    for i, acc in enumerate(avg_acc):
        name = load_dir.name.replace("acc", "avg_acc")
        name = "_".join(name.split("_")[:9])
        name = name.replace("_cw", "").replace("_ce", "") + f"_{i}"
        dir = load_dir.parent / name
        dir.mkdir()
        np.save(str(dir / "avg_acc.npy"), acc)
