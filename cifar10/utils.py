import scipy.io


def load_madry(load_dir, model_vars, model_type="plain"):
    w = scipy.io.loadmat(load_dir)
    mapping = {
        "A0": "conv2d/kernel:0",
        "A1": "conv2d_1/kernel:0",
        "A2": "conv2d_2/kernel:0",
        "A3": "conv2d_3/kernel:0",
        "A4": "conv2d_4/kernel:0",
        "A5": "conv2d_5/kernel:0",
        "A6": "conv2d_6/kernel:0",
        "A7": "conv2d_7/kernel:0",
        "A8": "dense/kernel:0",
        "A9": "dense/bias:0",
        "A10": "dense_1/kernel:0",
        "A11": "dense_1/bias:0",
    }
    if model_type == "plain":
        mapping.update(
            {
                "bA0": "conv2d/bias:0",
                "bA1": "conv2d_1/bias:0",
                "bA2": "conv2d_2/bias:0",
                "bA3": "conv2d_3/bias:0",
                "bA4": "conv2d_4/bias:0",
                "bA5": "conv2d_5/bias:0",
                "bA6": "conv2d_6/bias:0",
                "bA7": "conv2d_7/bias:0",
                "A8": "dense/kernel:0",
                "bA8": "dense/bias:0",
                "A9": "dense_1/kernel:0",
                "bA9": "dense_1/bias:0",
            }
        )
        del mapping["A10"]
        del mapping["A11"]
    for var_name in sorted(w.keys()):
        if not var_name.startswith("A"):
            continue
        var = w[var_name]
        if var.ndim == 2:
            var = var.squeeze()
        model_var_name = mapping[var_name]
        model_var = [v for v in model_vars if v.name == model_var_name]
        assert len(model_var) == 1
        model_var[0].assign(var)
