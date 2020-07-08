import functools
import itertools

import tensorflow as tf
import torch
from tensorflow.python.training import py_checkpoint_reader


def map_var_name(var_name):
    if var_name.endswith("kernel:0"):
        var_name = var_name.replace("kernel:0", "W")
    if var_name.endswith("bias:0"):
        var_name = var_name.replace("bias:0", "b")
    if var_name.endswith("beta:0"):
        var_name = var_name.replace("beta:0", "beta")
    if var_name.endswith("gamma:0"):
        var_name = var_name.replace("gamma:0", "gamma")
    if "dense" in var_name:
        var_name = var_name.replace("dense", "linear")
    if var_name.endswith("moving_mean:0"):
        var_name = var_name.replace("moving_mean:0", "mean/EMA")
    if var_name.endswith("moving_variance:0"):
        var_name = var_name.replace("moving_variance:0", "variance/EMA")
    if "shortcut/conv" in var_name:
        var_name = var_name.replace("shortcut/conv", "convshortcut")
    if "shortcut/bn" in var_name:
        var_name = var_name.replace("shortcut/bn", "convshortcut/bn")
    return var_name


def load_tsipras(load_from, model_vars):
    ckpt_manager = tf.train.CheckpointManager(tf.train.Checkpoint(),
                                              load_from,
                                              max_to_keep=3)
    ckpt_reader = py_checkpoint_reader.NewCheckpointReader(
        ckpt_manager.latest_checkpoint)
    all_ckpt_tensors = ckpt_reader.get_variable_to_shape_map()
    all_ckpt_tensors = {
        name: name.endswith("Momentum") or name.startswith("EMA")
        for name in all_ckpt_tensors
    }
    initialized_vars = {var.name: False for var in model_vars}
    for var in model_vars:
        var_name = var.name
        mapped_var_name = map_var_name(var_name)
        try:
            var_loaded_value = ckpt_reader.get_tensor(mapped_var_name)
            var.assign(var_loaded_value)
            initialized_vars[var_name] = True
            all_ckpt_tensors[mapped_var_name] = True
        except:
            print("Failed to find: {}".format(mapped_var_name))

    assert all([v for v in initialized_vars.values()]), "Failed to load model"
    print("Failed to find a matching variable ckpt -> model: {}".format(
        [name for name, v in all_ckpt_tensors.items() if not v]))


def map_var_name_pt(var_name):
    var_name = functools.reduce(
        lambda n, i: n.replace("backbone.layer%d.%d" %
                               (i[0], i[1]), "group%d/block%d" %
                               (i[0] - 1, i[1])),
        itertools.product(range(10), range(10)), var_name)
    var_name = var_name.replace("backbone.", "")
    var_name = functools.reduce(
        lambda n, i: n.replace("block%d.bn%d" %
                               (i[0], i[1]), "block%d/conv%d/bn" %
                               (i[0], i[1])),
        itertools.product(range(10), range(10)), var_name)
    var_name = functools.reduce(
        lambda n, i: n.replace("block%d.conv%d" %
                               (i[0], i[1]), "block%d/conv%d" % (i[0], i[1])),
        itertools.product(range(10), range(10)), var_name)
    # handle dense layers naming
    var_name = var_name.replace("fc.weight", "linear/W")
    var_name = var_name.replace("fc.bias", "linear/b")
    # handle convolutional layers naming
    var_name = functools.reduce(
        lambda n, i: n.replace("/conv%d.weight" % i, "/conv%d/W" % i),
        range(10), var_name)
    var_name = var_name.replace("conv1.weight", "conv0/W")
    var_name = var_name.replace(".downsample.0", "/convshortcut")
    var_name = var_name.replace(".downsample.1", "/convshortcut/bn")
    # handle batch normalization layers naming
    var_name = var_name.replace("bn1", "conv0/bn")
    var_name = var_name.replace("bn.weight", "bn/gamma")
    var_name = var_name.replace("bn.bias", "bn/beta")
    var_name = var_name.replace(".running_var", "/variance/EMA")
    var_name = var_name.replace(".running_mean", "/mean/EMA")
    # handle remaining naming issues
    var_name = var_name.replace(".weight", "/W")
    return var_name


def load_tsipras_pt(load_from, model_params):
    ckpt_manager = tf.train.CheckpointManager(tf.train.Checkpoint(),
                                              load_from,
                                              max_to_keep=3)
    ckpt_reader = py_checkpoint_reader.NewCheckpointReader(
        ckpt_manager.latest_checkpoint)
    all_ckpt_tensors = ckpt_reader.get_variable_to_shape_map()
    all_ckpt_tensors = {
        name: name.endswith("Momentum") or name.startswith("EMA")
        for name in all_ckpt_tensors
    }
    model_params = {
        key: value
        for key, value in model_params.items()
        if "num_batches_tracked" not in key
    }
    initialized_vars = {param_name: False for param_name in model_params}
    for param_name, param in model_params.items():
        mapped_param_name = map_var_name_pt(param_name)
        try:
            param_loaded_value = ckpt_reader.get_tensor(mapped_param_name)
            if param_loaded_value.ndim == 4:
                param_loaded_value = param_loaded_value.transpose(3, 2, 0, 1)
            elif param_loaded_value.ndim == 2:
                param_loaded_value = param_loaded_value.transpose(1, 0)
            param.data.copy_(torch.from_numpy(param_loaded_value))
            initialized_vars[param_name] = True
            all_ckpt_tensors[mapped_param_name] = True
        except:
            print("Failed to find: {}".format(mapped_param_name))

    assert all([v for v in initialized_vars.values()]), "Failed to load model"
    print("Failed to find a matching variable ckpt -> model: {}".format(
        [name for name, v in all_ckpt_tensors.items() if not v]))
