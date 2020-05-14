import numpy as np

test_model_thresholds = {
    "plain": {
        "l0": [5, 6, 8, 10, 15],
        "l1": np.linspace(2, 10, 5),
        "l2": np.linspace(0.5, 2.5, 5),
        "li": np.linspace(0.03, 0.11, 5)
    },
    "linf": {
        "l0": np.linspace(2, 10, 5, dtype=np.int32),
        "l1": np.linspace(2.5, 12.5, 5),
        "l2": np.linspace(1.0, 3.0, 5),
        "li": [0.2, 0.25, 0.3, 0.325, 0.35],
    },
    "l2": {
        "l0": np.linspace(5, 25, 5, dtype=np.int32),
        "l1": np.linspace(5, 20, 5),
        "l2": np.linspace(1.0, 3.0, 5),
        "li": np.linspace(0.05, 0.25, 5),
    }
}

test_thresholds = {"l0": [1, 3, 5], "l1": [], "l2": [], "li": []}
for model in test_model_thresholds.keys():
    thresholds = test_model_thresholds[model]
    for norm in thresholds:
        test_thresholds[norm].extend(thresholds[norm])

test_thresholds = {
    norm: sorted(list(set(thresholds)))
    for norm, thresholds in test_thresholds.items()
}
