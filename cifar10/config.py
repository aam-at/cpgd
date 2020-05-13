import numpy as np

test_model_thresholds = {
    "plain": {
        "l0": np.linspace(0, 15, 16),
        "l1": np.linspace(2, 10, 5),
        "l2": [0.1, 0.15, 0.2, 0.3, 0.4],
        "li": np.linspace(1, 3, 5) / 255.0
    },
    "linf": {
        "l0": np.linspace(0, 15, 16),
        "l1": np.linspace(5, 20, 5),
        "l2": np.linspace(0.25, 1.25, 5),
        "li": np.linspace(2, 10, 5) / 255.0
    },
    "l2": {
        "l0": np.linspace(0, 15, 16),
        "l1": np.linspace(3, 15, 5),
        "l2": np.linspace(0.25, 1.25, 5),
        "li": np.linspace(2, 10, 5) / 255.0
    }
}

test_thresholds = {"l0": [], "l1": [], "l2": [], "li": []}
for model in test_model_thresholds.keys():
    thresholds = test_model_thresholds[model]
    for norm in thresholds:
        test_thresholds[norm].extend(thresholds[norm])

test_thresholds = {
    norm: sorted(list(set(thresholds)))
    for norm, thresholds in test_thresholds.items()
}
