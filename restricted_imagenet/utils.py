import pickle

import numpy as np
import scipy.io
import tensorflow as tf
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

    print("Failed to find a matching variable ckpt -> model: {}".format(
        [name for name, v in all_ckpt_tensors.items() if not v]))
    print("Failed to find a matching variable model -> ckpt: {}".format(
        [name for name, v in initialized_vars.items() if not v]))
