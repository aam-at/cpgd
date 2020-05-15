import numpy as np

test_model_thresholds = {
    "plain": {
        "l0": [1, 3, 5, 8, 12],
        "l1": np.linspace(2, 10, 5, dtype=np.int32),
        "l2": np.round(np.linspace(0.5, 2.5, 5), 1),
        "li": np.round(np.linspace(0.03, 0.11, 5), 2)
    },
    "linf": {
        "l0": [1, 3, 5, 8, 12],
        "l1": np.round(np.linspace(2.5, 12.5, 5), 1),
        "l2": np.round(np.linspace(1.0, 3.0, 5), 1),
        "li": [0.2, 0.25, 0.3, 0.325, 0.35],
    },
    "l2": {
        "l0": np.linspace(5, 25, 5, dtype=np.int32),
        "l1": np.round(np.linspace(5, 20, 5), 2),
        "l2": np.round(np.linspace(1.0, 3.0, 5), 1),
        "li": np.round(np.linspace(0.05, 0.25, 5), 2),
    }
}

test_thresholds = {
    "l0": list(np.linspace(1, 25, 25)),
    "l1": [],
    "l2": [],
    "li": []
}
for model in test_model_thresholds.keys():
    thresholds = test_model_thresholds[model]
    for norm in thresholds:
        test_thresholds[norm].extend(thresholds[norm])

test_thresholds = {
    norm: sorted(list(set(thresholds)))
    for norm, thresholds in test_thresholds.items()
}
