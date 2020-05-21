import numpy as np

test_model_thresholds = {
    "plain": {
        "l0": np.linspace(10, 50, 5, dtype=np.int32),
        "l1": np.linspace(5, 49, 5, dtype=np.int32),
        "l2": np.round(np.linspace(0.2, 1, 5), 2),
        "li": np.round(np.linspace(0.25, 1.25, 5), 2) / 255.0
    },
    "linf": {
        "l0": [10, 30, 50, 80, 100],
        "l1": [15, 25, 40, 60, 100],
        "l2": np.linspace(1, 5, 5, dtype=np.int32),
        "li": np.linspace(2, 10, 5, dtype=np.int32) / 255.0
    },
    "l2": {
        "l0": np.linspace(50, 250, 5, dtype=np.int32),
        "l1": np.linspace(50, 250, 5, dtype=np.int32),
        "l2": np.linspace(2, 6, 5, dtype=np.int32),
        "li": np.linspace(2, 10, 5, dtype=np.int32) / 255.0
    }
}

test_thresholds = {
    "l0": list(np.linspace(1, 500, 500, dtype=np.int32)),
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
