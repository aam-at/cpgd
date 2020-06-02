from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .at import fast_gradient_perturbation
from .utils import (arctanh_reparametrize, li_metric, prediction,
                    random_targets, tanh_reparametrize, to_indexed_slices)

tfd = tfp.distributions


class FastGradientMethod(object):
    def __init__(self, model, epsilon=0.1, boxmin=None, boxmax=None):
        self.model = model
        self.epsilon = epsilon
        self.boxmin = boxmin
        self.boxmax = boxmax

    @tf.function
    def __call__(self, X, y_true=None):
        r = self.call(X, y_true)
        X_adv = X + r
        if self.boxmin is not None and self.boxmax is not None:
            X_adv = tf.clip_by_value(X_adv, self.boxmin, self.boxmax)
        return X_adv

    def call(self, X, y_true=None):
        r = fast_gradient_perturbation(self.model,
                                       X,
                                       y_true,
                                       epsilon=self.epsilon)
        return r


def _find_next_target(model,
                      X,
                      y_true,
                      random=False,
                      uniform=False,
                      label_smoothing=0.0,
                      epsilon=1e-8,
                      ord=2):
    """Find closest decision boundary as in Deepfool algorithm"""
    batch_size = X.get_shape()[0]
    batch_indices = tf.range(batch_size, dtype=y_true.dtype)
    ndims = X.get_shape().ndims
    y_true_idx = tf.stack([batch_indices, y_true], axis=1)

    if not random:
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            logits = model(X)
            logits_true = tf.gather_nd(logits, y_true_idx)
            f = tf.unstack(logits, axis=1)

        grad_labels = tape.gradient(logits_true, X)
        w = tf.stack([tape.gradient(f_i, X) - grad_labels for f_i in f],
                     axis=1)
        f = tf.stack([f_i - logits_true for f_i in f], axis=1)

        # select closest boundary
        reduce_ind = list(range(2, ndims + 1))
        if ord == 2:
            dist = tf.abs(f) / tf.sqrt(epsilon +
                                       tf.reduce_sum(w**2, axis=reduce_ind))
        else:
            dist = tf.abs(f) / tf.reduce_sum(epsilon + tf.abs(w),
                                             axis=reduce_ind)
        dist = tf.tensor_scatter_nd_update(dist, y_true_idx,
                                           np.inf * tf.ones(batch_size, 1))
        return tf.cast(tf.argmin(dist, axis=1), y_true.dtype)
    else:
        logits = model(X)
        num_classes = logits.get_shape()[-1]
        y_true_onehot = tf.one_hot(y_true, num_classes)
        return random_targets(num_classes, y_true_onehot)


class HighConfidenceAttack(object):
    def __init__(self,
                 model,
                 confidence=0.8,
                 max_iter=25,
                 over_shoot=0.02,
                 ord=2,
                 clip_dist=None,
                 attack_random=False,
                 attack_uniform=False,
                 epsilon=1e-8,
                 boxmin=None,
                 boxmax=None):
        assert ord == 2
        self.model = model
        self.confidence = confidence
        self.max_iter = max_iter
        self.over_shoot = over_shoot
        self.ord = ord
        self.clip_dist = clip_dist
        self.attack_random = attack_random
        self.attack_uniform = attack_uniform
        self.epsilon = epsilon
        self.boxmin = boxmin
        self.boxmax = boxmax

    def __call__(self, X, y_true=None, y_targ=None, C=None):
        if y_true is None:
            logits = self.model(X)
            y_true = prediction(logits)
        if y_targ is None:
            y_targ = _find_next_target(self.model,
                                       X,
                                       y_true,
                                       random=self.attack_random,
                                       uniform=self.attack_uniform)
        if C is None:
            C = self.confidence
        r = self.call(X, y_true, y_targ, C)
        X_adv = X + (1 + self.over_shoot) * r
        if self.boxmin is not None and self.boxmax is not None:
            X_adv = tf.clip_by_value(X_adv, self.boxmin, self.boxmax)
        return X_adv

    @tf.function
    def call(self, X, y_true, y_targ, C):
        epsilon = self.epsilon
        ndims = X.get_shape().ndims
        reduce_ind = list(range(1, ndims))
        batch_size = X.get_shape()[0]
        batch_indices = tf.range(batch_size, dtype=y_true.dtype)
        y_targ_idx = tf.stack([batch_indices, y_targ], axis=1)

        # initial perturbation
        X_adv = X
        f = tf.zeros((batch_size, ) + (1, ) * (ndims - 1), name='f')
        r = tf.zeros_like(X)
        for iteration in tf.range(self.max_iter):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(X_adv)
                logits_adv = self.model(X_adv)
                prob_adv = tf.nn.softmax(logits_adv)
                logprob_adv = tf.nn.log_softmax(logits_adv)

                prob_adv_t = tf.gather_nd(prob_adv, y_targ_idx)
                logprob_adv_t = tf.gather_nd(logprob_adv, y_targ_idx)
                # check if all examples are mistakes already
                is_high_confidence = tf.greater_equal(prob_adv_t, C)
                is_target_hit = tf.equal(prediction(logits_adv), y_targ)
                target_is_label = tf.equal(y_true, y_targ)
                selector = tf.logical_or(
                    tf.logical_and(is_target_hit, is_high_confidence),
                    target_is_label)
                if tf.reduce_all(selector):
                    break

                f = tf.reshape(tf.abs(logprob_adv_t - tf.math.log(C)),
                               f.get_shape())

            w = tape.gradient(logprob_adv_t, X_adv)
            w_norm = tf.sqrt(
                epsilon + tf.reduce_sum(w**2, axis=reduce_ind, keepdims=True))
            r_upd = (tf.math.sqrt(epsilon) + f / w_norm) * w / w_norm
            if self.clip_dist is not None:
                r_upd = tf.clip_by_norm(r_upd, self.clip_dist, axes=reduce_ind)

            r = tf.tensor_scatter_nd_add(
                r, tf.reshape(batch_indices[~selector], (-1, 1)),
                r_upd[~selector])
            X_adv = X + (1 + self.over_shoot) * r
            if self.boxmin is not None and self.boxmax is not None:
                r = (tf.clip_by_value(X_adv, self.boxmin, self.boxmax) -
                     X) / (1 + self.over_shoot)
        return r


class DeepFool(object):
    def __init__(self,
                 model,
                 max_iter=25,
                 over_shoot=0.02,
                 ord=2,
                 clip_dist=None,
                 epsilon=1e-8,
                 boxmin=None,
                 boxmax=None):
        self.model = model
        self.max_iter = max_iter
        self.over_shoot = over_shoot
        self.ord = ord
        self.clip_dist = clip_dist
        self.epsilon = epsilon
        self.boxmin = boxmin
        self.boxmax = boxmax

    def __call__(self, X, y_true=None):
        if y_true is None:
            logits = self.model(X)
            y_true = prediction(logits)
        r = self.call(X, y_true)
        X_adv = X + (1 + self.over_shoot) * r
        if self.boxmin is not None and self.boxmax is not None:
            X_adv = tf.clip_by_value(X_adv, self.boxmin, self.boxmax)
        return X_adv

    @tf.function
    def call(self, X, y_true):
        epsilon = self.epsilon
        ndims = X.get_shape().ndims
        reduce_ind = list(range(1, ndims))
        batch_size = X.get_shape()[0]
        batch_indices = tf.range(batch_size, dtype=y_true.dtype)
        y_true_idx = tf.stack([batch_indices, y_true], axis=1)

        # initial perturbation
        X_adv = X
        f = tf.zeros((batch_size, ))
        r = tf.zeros_like(X)
        for iteration in tf.range(self.max_iter):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(X_adv)
                logits_adv = self.model(X_adv)
                # check if all examples are mistakes already
                y_curr = prediction(logits_adv)
                is_correct = tf.equal(y_curr, y_true)
                if tf.reduce_all(tf.logical_not(is_correct)):
                    break

                # select next attack target if it is not given
                y_adv = _find_next_target(self.model, X_adv, y_true)
                y_adv_idx = tf.stack([batch_indices, y_adv], axis=1)

                # update perturbation
                logits_true = tf.gather_nd(logits_adv, y_true_idx)
                logits_adv = tf.gather_nd(logits_adv, y_adv_idx)
                f = logits_adv - logits_true
            w = tape.gradient(f, X_adv)
            w2_norm = tf.sqrt(epsilon + tf.reduce_sum(w**2, axis=reduce_ind))
            if self.ord == 2:
                dist = tf.abs(f) / w2_norm
            else:
                dist = tf.abs(f) / tf.reduce_sum(epsilon + tf.abs(w),
                                                 axis=reduce_ind)
            # avoid numerical instability and clip max value
            if self.clip_dist is not None:
                dist = tf.clip_by_value(dist, 0, self.clip_dist)
            if self.ord == 2:
                r_upd = w * tf.reshape(((dist + 1e-4) / w2_norm),
                                       (-1, ) + (1, ) * (ndims - 1))
            else:
                r_upd = tf.sign(w) * tf.reshape(dist,
                                                (-1, ) + (1, ) * (ndims - 1))
            r = tf.tensor_scatter_nd_add(
                r, tf.reshape(batch_indices[is_correct], (-1, 1)),
                r_upd[is_correct])
            X_adv = X + (1 + self.over_shoot) * r
            if self.boxmin is not None and self.boxmax is not None:
                r = (tf.clip_by_value(X_adv, self.boxmin, self.boxmax) -
                     X) / (1 + self.over_shoot)
        return r
