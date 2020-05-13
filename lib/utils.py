from __future__ import absolute_import, division, print_function

import inspect
import json
import logging
import os
import subprocess
from argparse import Namespace
from functools import partial
from pathlib import Path
from shutil import copyfile

import numpy as np
import six
import tensorflow as tf
from absl import flags
from absl.flags import DuplicateFlagError
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

FLAGS = flags.FLAGS


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __getattr__(self, key):
        try:
            value = self.__getitem__(key)
        except KeyError as exc:
            return None
        if isinstance(value, dict):
            value = AttributeDict(value)
        return value

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


class MetricsDictionary(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            dict.__setitem__(self, key, tf.keras.metrics.Mean(key))
            return self.__getitem__(key)


def flags_to_params(fls):
    return Namespace(**{k: f.value for k, f in fls.__flags.items()})


def import_klass_kwargs_as_flags(klass, prefix=''):
    for base_klass in klass.mro():
        import_kwargs_as_flags(base_klass.__init__, prefix)


def import_kwargs_as_flags(f, prefix=''):
    spec = inspect.getfullargspec(f)
    flag_defines = {
        str: flags.DEFINE_string,
        bool: flags.DEFINE_bool,
        int: flags.DEFINE_integer,
        float: flags.DEFINE_float
    }
    for index, (kwarg, kwarg_type) in enumerate(spec.annotations.items()):
        try:
            kwarg_default = spec.defaults[index]
        except:
            kwarg_default = spec.kwonlydefaults[kwarg]
        if kwarg_type not in flag_defines:
            logging.debug(f"Uknown {kwarg} type {kwarg_type}")
        else:
            arg_name = f"{prefix}{kwarg}"
            try:
                flag_defines[kwarg_type](arg_name, kwarg_default,
                                        f"{kwarg}")
            except DuplicateFlagError as e:
                logging.debug(e)


def prepare_dir(dir_path, subdir_name):
    base = os.path.join(dir_path, subdir_name)
    i = 0
    while True:
        name = base + str(i)
        try:
            os.makedirs(name)
            break
        except OSError:
            i += 1
    return name


class NanError(BaseException):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return "NanError: %s has nan value" % self.message


# data utils
def select_balanced_subset(X, y, num_classes=10, samples_per_class=10, seed=1):
    total_samples = num_classes * samples_per_class
    X_subset = np.zeros([total_samples] + list(X.shape[1:]), dtype=X.dtype)
    y_subset = np.zeros((total_samples, ), dtype=y.dtype)
    rng = np.random.RandomState(seed)
    for i in range(num_classes):
        yi_indices = np.where(y == i)[0]
        rng.shuffle(yi_indices)
        X_subset[samples_per_class * i:(i + 1) * samples_per_class,
                 ...] = X[yi_indices[:samples_per_class]]
        y_subset[samples_per_class * i:(i + 1) * samples_per_class] = i
    return X_subset, y_subset


def make_input_pipeline(ds, shuffle=True, batch_size=128):
    if shuffle:
        ds = ds.shuffle(10 * batch_size)
    return ds.batch(batch_size, drop_remainder=True).prefetch(1)


def pad_images(images, fltr):
    nimages = images.shape[0]
    padded_images = []
    for i in range(nimages):
        if fltr[i]:
            padded_images.append(
                np.pad(images[i], [(0, 0), (2, 2), (2, 2)],
                       'constant',
                       constant_values=1.0))
        else:
            padded_images.append(
                np.pad(images[i], [(0, 0), (2, 2), (2, 2)],
                       'constant',
                       constant_values=0.0))
    return np.vstack(padded_images)[:, np.newaxis, ...]


def save_images(images, path, data_format="NCHW", **kwargs):
    import torch
    from torchvision.utils import save_image
    if "nrow" not in kwargs:
        kwargs["nrow"] = int(np.sqrt(images.shape[0]))
    if data_format == "NHWC":
        images = np.transpose(images, (0, 3, 1, 2))
    save_image(torch.from_numpy(images), path, **kwargs)


def register_experiment_flags(working_dir="runs", seed=1):
    # experiment parameters
    flags.DEFINE_string("name", None, "name of the experiment")
    flags.DEFINE_integer("seed", seed, "experiment seed")
    flags.DEFINE_integer("data_seed", 1, "experiment seed")
    flags.DEFINE_string("working_dir", working_dir, "path to working dir")
    flags.DEFINE_string("chks_dir", "chks", "path to checks dir")
    flags.DEFINE_string("samples_dir", "samples", "path to samples dir")
    flags.DEFINE_string("git_revision", None, "git revision")


def setup_experiment(default_name, snapshot_files=None):
    from logging import FileHandler, Formatter, StreamHandler
    np.random.seed(FLAGS.seed)
    if tf.version.VERSION.startswith("1."):
        tf.compat.v1.set_random_seed(FLAGS.seed)
    else:
        tf.random.set_seed(FLAGS.seed)

    dict_values = {k: v.value for k, v in FLAGS._flags().items()}
    if FLAGS.name is None:
        FLAGS.name = default_name % dict_values
    FLAGS.git_revision = get_sha()
    FLAGS.working_dir = prepare_dir(FLAGS.working_dir, FLAGS.name)
    FLAGS.chks_dir = os.path.join(FLAGS.working_dir, FLAGS.chks_dir)
    FLAGS.samples_dir = os.path.join(FLAGS.working_dir, FLAGS.samples_dir)
    Path(FLAGS.chks_dir).mkdir()
    Path(FLAGS.samples_dir).mkdir()

    # configure logging
    logger = logging.getLogger()
    [logger.removeHandler(handler) for handler in logger.handlers]
    file_hndl = FileHandler(os.path.join(FLAGS.working_dir, 'tensorflow.log'))
    file_hndl.setLevel(logging.DEBUG)
    logger.addHandler(file_hndl)
    cmd_hndl = StreamHandler()
    cmd_hndl.setLevel(logging.INFO)
    cmd_hndl.setFormatter(Formatter('%(message)s'))
    logger.addHandler(cmd_hndl)
    logger.setLevel(logging.DEBUG)

    # print config
    train_params = json.dumps({k: v.value
                               for k, v in FLAGS._flags().items()},
                              sort_keys=True)
    logging.info(train_params)

    if snapshot_files is not None:
        for snapshot_file in snapshot_files:
            copyfile(
                snapshot_file,
                os.path.join(FLAGS.working_dir,
                             os.path.basename(snapshot_file)))


# tensorflow utils
def create_optimizer(opt, lr, **kwargs):
    config = {'learning_rate': lr}
    config.update(kwargs)
    return tf.keras.optimizers.get({'class_name': opt, 'config': config})


def reset_optimizer(opt):
    [var.assign(tf.zeros_like(var)) for var in opt.variables()]


def create_lr_schedule(schedule, **kwargs):
    if schedule == 'constant':
        lr = kwargs['learning_rate']
    elif schedule == 'linear':
        lr = LinearDecay(**kwargs)
    return lr


def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset_states()


def log_metrics(metrics, header=None, level=logging.INFO, throw_on_nan=False):
    str_bfr = six.StringIO()
    if header is not None:
        str_bfr.write(header)
    for metric_name, metric_value in metrics.items():
        metric_value = metric_value.result()
        if np.isnan(metric_value):
            if throw_on_nan:
                raise NanError(metric_name)
            else:
                metric_value = -1
        str_bfr.write(" {}: {:.6f},".format(metric_name, metric_value))
    logging.log(level, str_bfr.getvalue()[:-1])


def log_tensorboard_metrics(metrics, prefix=None, step=None):
    for metric_name, metric_value in metrics.items():
        metric_value = metric_value.result()
        tf.summary.scalar(f"{prefix}{metric_name}", metric_value, step)


def kl_with_logits(q_logits, p_logits):
    q = tf.nn.softmax(q_logits)
    q_log = tf.nn.log_softmax(q_logits)
    p_log = tf.nn.log_softmax(p_logits)
    return tf.reduce_sum(q * (q_log - p_log), axis=1)


def Hv_finite(f, g, x, v, xi=1e-6):
    """Multiply the Hessian of `f` wrt `x` by `v` (using finite difference).
    """
    v *= xi
    with tf.GradientTape(persistent=True) as tape:
        # First backprop
        tape.watch(v)
        y = f(x)
        y_v = f(x + v)
        loss = g(y, y_v)
    return tape.gradient(loss, v) / xi


def power_iteration(Ax, x0, num_iterations):
    xk = l2_normalize(x0)
    for i in range(num_iterations):
        xk = Ax(xk)
        xk = l2_normalize(xk)
        xk = tf.stop_gradient(xk)
    return xk


class ConstantDecay(LearningRateSchedule):
    def __init__(self, learning_rate, name=None):
        super(ConstantDecay, self).__init__()
        self.learning_rate = learning_rate
        self.name = name

    def __call__(self, step):
        return self.learning_rate

    def get_config(self):
        return {"learning_rate": self.learning_rate, "name": self.name}


class LinearDecay(LearningRateSchedule):
    def __init__(self,
                 initial_learning_rate,
                 minimal_learning_rate,
                 decay_steps,
                 name=None):
        super(LinearDecay, self).__init__()
        assert initial_learning_rate > minimal_learning_rate, (
            initial_learning_rate, minimal_learning_rate)
        self.initial_learning_rate = initial_learning_rate
        self.minimal_learning_rate = minimal_learning_rate
        self.decay_steps = decay_steps
        self.name = name

    def __call__(self, step):
        initial_learning_rate = tf.convert_to_tensor(
            self.initial_learning_rate, name="initial_learning_rate")
        dtype = initial_learning_rate.dtype
        minimal_learning_rate = tf.cast(self.minimal_learning_rate, dtype)
        decay_steps = tf.cast(self.decay_steps, dtype)

        global_step_recomp = tf.cast(step, dtype)
        p = global_step_recomp / decay_steps

        assert_op = tf.Assert(decay_steps >= global_step_recomp, [step])
        with tf.control_dependencies([assert_op]):
            return (minimal_learning_rate +
                    (initial_learning_rate - minimal_learning_rate) * (1 - p))

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "minimal_learning_rate": self.minimal_learning_rate,
            "decay_steps": self.decay_steps,
            "name": self.name
        }


def l0_metric(x, axes=None, keepdims=False):
    if axes is None:
        axes = list(range(1, x.shape.ndims))
    x = tf.convert_to_tensor(x, name="x")
    return tf.reduce_sum(tf.cast(tf.abs(x) > 0, x.dtype),
                         axes,
                         keepdims=keepdims)


def l0_pixel_metric(u, channel_dim=-1, keepdims=False):
    u_c = tf.reduce_max(tf.abs(u), axis=channel_dim)
    return l0_metric(u_c, keepdims=keepdims)


def li_metric(x, axes=None, keepdims=False):
    if axes is None:
        axes = list(range(1, x.shape.ndims))
    x = tf.convert_to_tensor(x, name="x")
    return tf.reduce_max(tf.abs(x), axes, keepdims=keepdims)


def li_normalize(d, eps=1e-6, axes=None):
    if axes is None:
        axes = list(range(1, d.shape.ndims))
    d = tf.convert_to_tensor(d, name="d")
    d_li = li_metric(d, axes, keepdims=True)
    return d / (d_li + eps)


def l1_metric(x, axes=None, keepdims=False):
    """L1 metric
    """
    if axes is None:
        axes = list(range(1, x.shape.ndims))
    x = tf.convert_to_tensor(x, name="x")
    return tf.reduce_sum(tf.abs(x), axes, keepdims=keepdims)


def l1_normalize(d, axes=None):
    """L1 normalization
    """
    if axes is None:
        axes = list(range(1, d.shape.ndims))
    d = tf.convert_to_tensor(d, name="d")
    d_l1 = l1_metric(d, axes, keepdims=True)
    return d / d_l1


def l2_metric(x, epsilon=1e-12, axes=None, keepdims=False):
    """Stable l2 normalization
    """
    if axes is None:
        axes = list(range(1, x.shape.ndims))
    x = tf.convert_to_tensor(x, name="x")
    square_sum = tf.reduce_sum(tf.square(x), axes, keepdims=keepdims)
    return tf.math.sqrt(epsilon + square_sum)


def l2_normalize(d, axes=None, epsilon=1e-12):
    """Stable l2 normalization
    """
    if axes is None:
        axes = list(range(1, d.shape.ndims))
    d = tf.convert_to_tensor(d, name="d")
    d /= (epsilon + tf.reduce_max(tf.abs(d), axes, keepdims=True))
    d_square_sum = tf.reduce_sum(tf.square(d), axes, keepdims=True)
    d_inv_norm = tf.math.rsqrt(epsilon + d_square_sum)
    return tf.multiply(d, d_inv_norm)


def get_acc_for_lp_threshold(model, image_o, image_m, label, lp, threshold):
    image_th = tf.where(tf.reshape(lp <= threshold, (-1, 1, 1, 1)), image_m,
                        image_o)
    logits_th = model(image_th)
    acc_th = tf.keras.metrics.sparse_categorical_accuracy(label, logits_th)
    return acc_th


def entropy(l):
    p = tf.nn.softmax(l)
    lp = tf.nn.log_softmax(l)
    entropy = -tf.reduce_sum(lp * p, axis=-1)
    return entropy


def spectral_norm(inputs,
                  u_var,
                  power_iteration_rounds=1,
                  soft=False,
                  epsilon=1e-12,
                  training=True):
    """Performs Spectral Normalization on a weight tensor.

    Details of why this is helpful for GAN's can be found in "Spectral
    Normalization for Generative Adversarial Networks", Miyato T. et al., 2018.
    [https://arxiv.org/abs/1802.05957].

    Returns:
        The normalized weight tensor.
    """
    if len(inputs.shape) < 2:
        raise ValueError(
            "Spectral norm can only be applied to multi-dimensional tensors")
    # The paper says to flatten convnet kernel weights from (C_out, C_in, KH, KW)
    # to (C_out, C_in * KH * KW). Our Conv2D kernel shape is (KH, KW, C_in, C_out)
    # so it should be reshaped to (KH * KW * C_in, C_out), and similarly for other
    # layers that put output channels as last dimension. This implies that w
    # here is equivalent to w.T in the paper.
    w = tf.reshape(inputs, (-1, inputs.shape[-1]))

    # Choose whether to persist the first left or first right singular vector.
    # As the underlying matrix is PSD, this should be equivalent, but in practice
    # the shape of the persisted vector is different. Here one can choose whether
    # to maintain the left or right one, or pick the one which has the smaller
    # dimension. We use the same variable for the singular vector if we switch
    # from normal weights to EMA weights.
    u = u_var

    # Use power iteration method to approximate the spectral norm.
    # The authors suggest that one round of power iteration was sufficient in the
    # actual experiment to achieve satisfactory performance.
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of matrix `w`.
        v = tf.math.l2_normalize(tf.matmul(tf.transpose(w), u),
                                 epsilon=epsilon)
        u = tf.math.l2_normalize(tf.matmul(w, v), epsilon=epsilon)

    # The authors of SN-GAN chose to stop gradient propagating through u and v
    # and we maintain that option.
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    norm_value = tf.matmul(tf.matmul(tf.transpose(u), w), v)
    norm_value.shape.assert_is_fully_defined()
    norm_value.shape.assert_is_compatible_with([1, 1])
    if soft:
        norm_value = tf.maximum(1.0, norm_value)

    # Update the approximation.
    if training:
        with tf.control_dependencies([u_var.assign(u)]):
            w_normalized = w / norm_value
    else:
        w_normalized = w / norm_value
    # Deflate normalized weights to match the unnormalized tensor.
    w_tensor_normalized = tf.reshape(w_normalized, inputs.shape)
    return w_tensor_normalized


def spectral_norm_conv2d(inputs,
                         u_var,
                         strides,
                         padding,
                         data_format="channels_last",
                         power_iteration_rounds=1,
                         soft=False,
                         epsilon=1e-12,
                         training=True):
    # Paper: https://arxiv.org/pdf/1811.07457.pdf
    assert len(inputs.shape) == 4
    w = inputs
    u = u_var
    conv2d = partial(tf.nn.conv2d,
                     strides=strides,
                     padding=padding,
                     data_format=data_format)
    conv2d_t = partial(tf.nn.conv2d_transpose,
                       strides=strides,
                       padding=padding,
                       data_format=data_format)
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of linear operator `w`.
        v = tf.math.l2_normalize(conv2d(u, w), epsilon=epsilon)
        u = tf.math.l2_normalize(conv2d_t(v, w, u.shape), epsilon=epsilon)
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    norm_value = tf.reduce_sum(tf.multiply(conv2d(u, w), v))
    norm_value.shape.assert_is_fully_defined()
    if soft:
        norm_value = tf.maximum(1.0, norm_value)

    # Update the approximation.
    if training:
        with tf.control_dependencies([u_var.assign(u)]):
            w_normalized = w / norm_value
    else:
        w_normalized = w / norm_value
    return w_normalized


def spectral_norm_conv2d_transpose(inputs,
                                   u_var,
                                   v_shape,
                                   strides,
                                   padding,
                                   data_format="channels_last",
                                   power_iteration_rounds=1,
                                   soft=False,
                                   epsilon=1e-12,
                                   training=True):
    assert len(inputs.shape) == 4
    w = inputs
    u = u_var
    conv2d = partial(tf.nn.conv2d,
                     strides=strides,
                     padding=padding,
                     data_format=data_format)
    conv2d_t = partial(tf.nn.conv2d_transpose,
                       strides=strides,
                       padding=padding,
                       data_format=data_format)
    for _ in range(power_iteration_rounds):
        # `v` approximates the first right singular vector of linear operator `w`.
        v = tf.math.l2_normalize(conv2d_t(u, w, v_shape), epsilon=epsilon)
        u = tf.math.l2_normalize(conv2d(v, w), epsilon=epsilon)
    u = tf.stop_gradient(u)
    v = tf.stop_gradient(v)

    norm_value = tf.reduce_sum(tf.multiply(conv2d_t(u, w, v_shape), v))
    norm_value.shape.assert_is_fully_defined()
    if soft:
        norm_value = tf.maximum(1.0, norm_value)

    # Update the approximation.
    if training:
        with tf.control_dependencies([u_var.assign(u)]):
            w_normalized = w / norm_value
    else:
        w_normalized = w / norm_value
    return w_normalized


def compute_margin(onehot_labels, logits, delta=0.0):
    labels_rank = onehot_labels.get_shape().ndims
    logits_rank = logits.get_shape().ndims
    if labels_rank == logits_rank - 1:
        onehot_labels = tf.one_hot(onehot_labels, logits.get_shape()[1])
    elif labels_rank != logits_rank:
        raise TypeError('rank mismatch between onehot_labels and logits')
    negative_inf = -np.inf * tf.ones_like(logits)
    corrects = tf.reduce_max(
        tf.where(tf.equal(onehot_labels, 1), logits, negative_inf), 1)
    rest = tf.reduce_max(
        tf.where(tf.equal(onehot_labels, 0), logits, negative_inf), 1)
    loss = tf.nn.relu(rest - corrects + delta)


def multiclass_margin(onehot_labels, logits, delta=1.0):
    margin = compute_margin(onehot_labels, logits)
    loss = tf.nn.relu(delta - margin)
    return loss


def random_targets(num_labels,
                   label_onehot=None,
                   logits=None,
                   num_samples=None,
                   dtype=tf.int32):
    if num_samples is None:
        assert label_onehot is not None
        num_samples = label_onehot.get_shape()[0]
    if label_onehot is None:
        return tf.random.uniform([num_samples],
                                 0,
                                 maxval=num_labels,
                                 dtype=tf.int32)
    else:
        if label_onehot.get_shape().ndims != 2:
            label_onehot = tf.one_hot(label_onehot, num_labels)
        if logits is None:
            logits = tf.ones_like(label_onehot, dtype=tf.float32)
        logits = tf.where(
            tf.cast(label_onehot, dtype=tf.bool),
            -np.inf * tf.ones_like(label_onehot, dtype=tf.float32), logits)
        return tf.reshape(
            tf.random.categorical(logits, num_samples=1, dtype=dtype), (-1, ))


def jacobian(y, x, pack_axis=1):
    jac = [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y, axis=1)]
    return tf.stack(jac, axis=pack_axis)


def get_lr_decay(lr_decay):
    def no_decay(base, *args):
        return base

    def linear_decay(base, end, pct):
        return base + pct * (end - base)

    def exp_decay(base, end, pct):
        return base * (end / base)**pct

    if lr_decay == 'linear':
        return linear_decay
    elif lr_decay == 'exp':
        return exp_decay
    elif lr_decay == 'no':
        return no_decay
    else:
        raise ValueError(lr_decay)


def tanh_reparametrize(x, boxplus=0.5, boxmul=0.5):
    return tf.tanh(x) * boxmul + boxplus


def arctanh_reparametrize(x, boxplus=0.5, boxmul=0.5):
    return tf.math.atanh((x - boxplus) / boxmul * 0.999999)


def prediction(prob, name='predictions'):
    return tf.cast(tf.argmax(prob, axis=-1), tf.int64, name=name)


def compute_norms(x, y):
    x = tf.reshape(x, (x.get_shape()[0], -1))
    y = tf.reshape(y, (y.get_shape()[0], -1))
    l2 = tf.norm(x - y, axis=1)
    l2_norm = l2 / tf.norm(x, axis=1)
    return l2, l2_norm


def to_indexed_slices(values, indices, mask=None):
    if mask is not None:
        return tf.IndexedSlices(values[mask], indices[mask])
    else:
        return tf.IndexedSlices(values, indices)


def add_default_end_points(end_points):
    logits = end_points['logits']
    predictions = prediction(logits)
    prob = tf.nn.softmax(logits, name='prob')
    log_prob = tf.nn.log_softmax(logits, name='log_prob')
    conf = tf.reduce_max(prob, axis=1)
    end_points.update({
        'pred': predictions,
        'prob': prob,
        'log_prob': log_prob,
        'conf': conf
    })
    return end_points


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw


def get_cls_pos_to_kw_map(cls):
    pos_to_kw = get_pos_to_kw_map(cls.__init__)
    for base_cls in cls.__bases__:
        # ignore 'self' argument
        for kw in get_cls_pos_to_kw_map(base_cls).values():
            i = len(pos_to_kw)
            if kw not in pos_to_kw.values():
                pos_to_kw[i] = kw
    return pos_to_kw


def change_default_args(layer_class, **kwargs):
    def layer_defaults(*args, **kw):
        pos_to_kw = get_cls_pos_to_kw_map(layer_class)
        kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
        for key, val in kwargs.items():
            if key not in kw and kw_to_pos[key] > len(args):
                kw[key] = val
        return layer_class(*args, **kw)

    return layer_defaults


# git utils
def get_sha(repo='.'):
    """
    Grabs the current SHA-1 hash of the given directory's git HEAD-revision.
    The output of this is equivalent to calling git rev-parse HEAD.

    Be aware that a missing git repository will make this return an error message,
    which is not a valid hash.
    """
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.decode('ascii').strip()
