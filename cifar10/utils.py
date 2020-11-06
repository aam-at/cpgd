import numpy as np
import scipy.io
import tensorflow as tf
import torch
from tensorflow.python.training import py_checkpoint_reader


def load_madry(load_dir, model_vars, model_type="plain", sess=None):
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
        if var_name.startswith("__"):
            continue
        var = w[var_name]
        if var.ndim == 2:
            var = var.squeeze()
        model_var_name = mapping[var_name]
        if "bias" in model_var_name and model_type == "plain":
            var = -var
        model_var = [v for v in model_vars if v.name == model_var_name]
        assert len(model_var) == 1
        if sess:
            sess.run(model_var[0].assign(var))
        else:
            model_var[0].assign(var)


def convert_madry_official_to_mat(load_from, save_to):
    ckpt_manager = tf.train.CheckpointManager(tf.train.Checkpoint(),
                                              load_from,
                                              max_to_keep=3)
    ckpt_reader = py_checkpoint_reader.NewCheckpointReader(
        ckpt_manager.latest_checkpoint)
    all_ckpt_tensors = ckpt_reader.get_variable_to_shape_map()
    all_ckpt_tensors = [name for name in all_ckpt_tensors
                        if not name.endswith("Momentum")]
    all_ckpt_tensors = sorted(all_ckpt_tensors)
    mdict = {}
    for var_name in all_ckpt_tensors:
        var_loaded_value = ckpt_reader.get_tensor(var_name)
        mdict[var_name] = var_loaded_value
    scipy.io.savemat(save_to, mdict)


def map_var_name(var_name):
    var_name = var_name.replace(":0", "")
    if var_name.endswith("kernel"):
        var_name = var_name.replace("kernel", "DW")
    if var_name.endswith("bias"):
        var_name = var_name.replace("bias", "biases")
    if "dense" in var_name:
        var_name = var_name.replace("dense", "logit")
    if var_name.startswith("conv0"):
        var_name = var_name.replace("conv0", "input/init_conv")
    if var_name.startswith("group"):
        group_index = index = int(var_name[5])
        var_name = var_name.replace("group%d/" % index, "unit_%d_" % (index + 1))
    else:
        group_index = -1
    if "block" in var_name:
        block_index = index = int(var_name[var_name.find("block") + 5])
        var_name = var_name.replace("block%d" % index, "%d" % (index))
    if "conv1/bn" in var_name:
        if group_index == 0 and block_index == 0:
            var_name = var_name.replace("conv1", "shared_activation")
        else:
            var_name = var_name.replace("conv1", "residual_only_activation")
    if "conv" in var_name and group_index != -1:
        conv_index = index = int(var_name[var_name.find("conv") + 4])
        var_name = var_name.replace("conv%d" % index, "sub%d/conv%d" % (index, index))
    else:
        conv_index = -1
    if "bn" in var_name:
        var_name = var_name.replace("bn", "BatchNorm")
        if conv_index > 1:
            var_name = var_name.replace("conv%d/" % index, "")

    return var_name


def load_madry_official(load_dir, model_vars, sess=None):
    w = scipy.io.loadmat(load_dir)
    initialized_vars = {var.name: False for var in model_vars}
    for var in model_vars:
        var_name = var.name
        mapped_var_name = map_var_name(var_name)
        try:
            var_loaded_value = w[mapped_var_name]
            var_loaded_value = np.squeeze(var_loaded_value)
            if sess:
                sess.run(var.assign(var_loaded_value))
            else:
                var.assign(var_loaded_value)
            initialized_vars[var_name] = True
        except:
            print("Failed to find: {}".format(mapped_var_name))
    print("Failed to find a matching variable .mat -> model: {}".format(
        [map_var_name(name) for name, v in initialized_vars.items() if not v]))


def load_madry_pt(load_from, model_params, model_type="plain"):
    w = scipy.io.loadmat(load_from)
    if model_type != "plain":
        names = [
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "A6",
            "A7",
            "A8",
            "A9",
            "A10",
            "A11",
        ]
    else:
        names = [
            "A0", "bA0",
            "A1", "bA1",
            "A2", "bA2",
            "A3", "bA3",
            "A4", "bA4",
            "A5", "bA5",
            "A6", "bA6",
            "A7", "bA7",
            "A8", "bA8",
            "A9", "bA9"
        ]
    t = list(model_params)
    for name, model_param in zip(names, t):
        load_var = w[name]
        if load_var.ndim == 2:
            load_var = load_var.transpose()
            load_var = load_var.squeeze()
        elif load_var.ndim == 4:
            load_var = load_var.transpose(3, 2, 0, 1)
        if load_var.ndim == 1 and model_type == "plain":
            load_var = -load_var
        model_param_shape = model_param.detach().numpy().shape
        load_var_shape = load_var.shape
        assert model_param_shape == load_var_shape
        model_param.data.copy_(torch.from_numpy(load_var))
