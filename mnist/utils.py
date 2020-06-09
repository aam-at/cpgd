import pickle

import numpy as np
import scipy.io
import tensorflow as tf
import torch
from tensorflow.python.training import py_checkpoint_reader


def load_madry(load_from, model_vars, sess=None):
    w = scipy.io.loadmat(load_from)
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
        if sess:
            sess.run(model_var[0].assign(var))
        else:
            model_var[0].assign(var)


def load_madry_official(load_from, model_vars):
    ckpt_manager = tf.train.CheckpointManager(tf.train.Checkpoint(),
                                              load_from,
                                              max_to_keep=3)
    ckpt_reader = py_checkpoint_reader.NewCheckpointReader(
        ckpt_manager.latest_checkpoint)
    mapping = {
        "conv2d/kernel:0": "Variable",
        "conv2d/bias:0": "Variable_1",
        "conv2d_1/kernel:0": "Variable_2",
        "conv2d_1/bias:0": "Variable_3",
        "dense/kernel:0": "Variable_4",
        "dense/bias:0": "Variable_5",
        "dense_1/kernel:0": "Variable_6",
        "dense_1/bias:0": "Variable_7"
    }
    initialized_vars = {var.name: False for var in model_vars}
    for model_var in model_vars:
        model_var_name = model_var.name
        try:
            var_loaded_value = ckpt_reader.get_tensor(mapping[model_var_name])
            model_var.assign(var_loaded_value)
        except:
            print("Failed to find: {}".format(model_var_name))


def load_madry_pt(load_from, model_params):
    w = scipy.io.loadmat(load_from)
    names = [
        "A0",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
    ]
    for load_var, model_param in zip([w[n] for n in names], model_params):
        if load_var.ndim == 2:
            load_var = load_var.transpose()
            load_var = load_var.squeeze()
        elif load_var.ndim == 4:
            load_var = load_var.transpose(3, 2, 0, 1)
        model_param_shape = model_param.detach().numpy().shape
        load_var_shape = load_var.shape
        assert model_param_shape == load_var_shape
        model_param.data.copy_(torch.from_numpy(load_var))


def convert_madry_official_to_mat(load_from, save_to):
    mapping = {
        "Variable": "A0",
        "Variable_1": "A1",
        "Variable_2": "A2",
        "Variable_3": "A3",
        "Variable_4": "A4",
        "Variable_5": "A5",
        "Variable_6": "A6",
        "Variable_7": "A7"
    }
    ckpt_manager = tf.train.CheckpointManager(tf.train.Checkpoint(),
                                              load_from,
                                              max_to_keep=3)
    ckpt_reader = py_checkpoint_reader.NewCheckpointReader(
        ckpt_manager.latest_checkpoint)
    mdict = {}
    for var_name in mapping.keys():
        var_loaded_value = ckpt_reader.get_tensor(var_name)
        mdict[mapping[var_name]] = var_loaded_value
    scipy.io.savemat(save_to, mdict)


def load_trades(load_from, model_vars):
    with open(load_from, 'rb') as handle:
        state = pickle.load(handle)
    for t, p in zip(model_vars, state.values()):
        if t.shape.ndims == 4:
            t.assign(np.transpose(p, (2, 3, 1, 0)))
        elif t.shape.ndims == 2:
            t.assign(p.transpose((1, 0)))
        else:
            t.assign(p)
