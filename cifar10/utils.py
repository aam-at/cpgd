import scipy.io


def load_madry(load_dir, model_vars):
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    w = scipy.io.loadmat(load_dir)
    mapping = {
        "A0": "conv2d/kernel:0",
        "A1": "conv2d/bias:0",
        "A2": "conv2d_1/kernel:0",
        "A3": "conv2d_1/bias:0",
        "A4": "dense/kernel:0",
        "A5": "dense/bias:0",
        "A6": "dense_1/kernel:0",
        "A7": "dense_1/bias:0"
    }
    for var_name in w.keys():
        if not var_name.startswith("A"):
            continue
        var = w[var_name]
        if var.ndim == 2:
            var = var.squeeze()
        model_var_name = mapping[var_name]
        model_var = [v for v in model_vars if v.name == model_var_name]
        assert len(model_var) == 1
        model_var[0].assign(var)
