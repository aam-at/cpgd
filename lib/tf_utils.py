from __future__ import absolute_import, division, print_function

import inspect

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import \
    LearningRateSchedule


class MetricsDictionary(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            dict.__setitem__(self, key, tf.keras.metrics.Mean(key))
            return self.__getitem__(key)


def make_input_pipeline(ds, shuffle=True, batch_size=128):
    if shuffle:
        ds = ds.shuffle(10 * batch_size)
    return ds.batch(batch_size, drop_remainder=True).prefetch(1)


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


def jacobian(y, x, pack_axis=1):
    jac = [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y, axis=1)]
    return tf.stack(jac, axis=pack_axis)


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


def l1_metric(x, axes=None, keepdims=False):
    """L1 metric
    """
    if axes is None:
        axes = list(range(1, x.shape.ndims))
    x = tf.convert_to_tensor(x, name="x")
    return tf.reduce_sum(tf.abs(x), axes, keepdims=keepdims)


def l2_metric(x, axes=None, keepdims=False):
    """Stable l2 normalization
    """
    if axes is None:
        axes = list(range(1, x.shape.ndims))
    x = tf.convert_to_tensor(x, name="x")
    square_sum = tf.reduce_sum(tf.square(x), axes, keepdims=keepdims)
    return tf.math.sqrt(square_sum)


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


def entropy(l):
    p = tf.nn.softmax(l)
    lp = tf.nn.log_softmax(l)
    entropy = -tf.reduce_sum(lp * p, axis=-1)
    return entropy


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
    return corrects - rest


def multiclass_margin(labels=None, logits=None, delta=0.0):
    margin = compute_margin(labels, logits)
    loss = tf.nn.relu(margin)
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


def prediction(prob, name='predictions'):
    return tf.cast(tf.argmax(prob, axis=-1), tf.int64, name=name)


def dist_matrix(a):
    """Compute distance matrix (combines scipy pdist and squareform)
    """
    with tf.control_dependencies([tf.assert_rank(a, 2)]):
        r = tf.reduce_sum(a * a, axis=1)
    r = tf.reshape(r, (-1, 1))
    D = r - 2 * tf.matmul(a, tf.transpose(a)) + tf.transpose(r)
    return D


def to_indexed_slices(values, indices, mask=None):
    if mask is not None:
        return tf.IndexedSlices(values[mask], indices[mask])
    else:
        return tf.IndexedSlices(values, indices)


def limit_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            raise


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


# tensorflow layer configuration utils
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
