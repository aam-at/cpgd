from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .at import fast_gradient_perturbation
from .utils import (arctanh_reparametrize, prediction,
                    random_targets, tanh_reparametrize, to_indexed_slices, li_metric)

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


def get_upper_bound_for_confidence(confidence, num_classes=10):
    assert np.all(0 <= confidence) and np.all(confidence <= 1.0)
    return np.log(confidence * (num_classes - 1) / (1 - confidence))


def get_lower_bound_for_confidence(confidence):
    assert np.all(0 <= confidence) and np.all(confidence <= 1.0)
    return np.log((1 - confidence) / confidence)


class CWL2(object):
    BINARY_SEARCH_STEPS = 9 # number of times to adjust the constant with binary search
    MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
    ABORT_EARLY = True      # if we stop improving, abort gradient descent early
    LEARNING_RATE = 1e-2    # larger values converge faster to less accurate results
    TARGETED = True         # should we target one specific class? or just be wrong?
    CONFIDENCE = 0          # how strong the adversarial example should be
    INITIAL_CONST = 1e-2    # the initial constant c to pick as a first guess

    def __init__(self,
                 model,
                 batch_size=1,
                 confidence=CONFIDENCE,
                 targeted=TARGETED,
                 learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST,
                 lower_bound=0,
                 upper_bound=1e6,
                 boxmin=0.0,
                 boxmax=1.0):
        """The L_2 optimization attack.

        This attack is the most efficient and should be used as the primary
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces
          examples that are farther away, but more strongly classified as
          adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False
          otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller
          values produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence.
        max_iterations: The maximum number of iterations. Larger values are
          more accurate; setting too small will require a large learning rate
          and will produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets
          stuck.
        initial_const: The initial tradeoff-constant to use to tune the
          relative importance of distance and confidence. If
          binary_search_steps is large, the initial constant is not important.
        boxmin: Minimum pixel value (default 0.0).
        boxmax: Maximum pixel value (default 1.0).
        """
        super(CWL2, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        self.CONFIDENCE = confidence
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.lower_bound_const = lower_bound
        self.upper_bound_const = upper_bound

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.

        self.built = False

    def build(self, inputs_shape):
        X_shape, y_shape = inputs_shape
        batch_size = X_shape[0]
        assert y_shape.ndims == 2
        self.r0 = tf.keras.initializers.get("zeros")
        self.r = tf.Variable(self.r0(X_shape), trainable=True, name="cwl2_r")
        self.lower_bound = tf.Variable(tf.zeros(batch_size),
                                       trainable=False,
                                       name="lb")
        self.const = tf.Variable(tf.zeros(batch_size),
                                 trainable=False,
                                 name="C")
        self.upper_bound = tf.Variable(tf.zeros(batch_size),
                                       trainable=False,
                                       name="ub")
        self.bestl2 = tf.Variable(tf.zeros(batch_size),
                                  trainable=False,
                                  name="best_l2")
        self.bestscore = tf.Variable(tf.zeros(batch_size),
                                     trainable=False,
                                     name="best_sc")
        self.o_bestl2 = tf.Variable(tf.zeros(batch_size),
                                    trainable=False,
                                    name="o_best_l2")
        self.o_bestattack = tf.Variable(tf.zeros(X_shape),
                                        trainable=False,
                                        name="o_bset_xhat")
        self.optimizer = tf.keras.optimizers.Adam(self.LEARNING_RATE)
        self.built = True

    def __call__(self, X, y_onehot):
        if not self.built:
            inputs_shapes = list(map(lambda x: x.shape, [X, y_onehot]))
            self.build(inputs_shapes)
        X_hat = self.call(X, y_onehot)
        return X_hat

    def _call(self, X, y_onehot):
        batch_size = self.batch_size
        batch_indices = self.batch_indices
        X_atanh = arctanh_reparametrize(X, self.boxplus, self.boxmul)

        ## get variables
        r = self.r
        r0 = self.r0
        optimizer = self.optimizer
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        const = self.const
        o_bestl2 = self.o_bestl2
        o_bestattack = self.o_bestattack
        bestl2 = self.bestl2
        bestscore = self.bestscore

        ## reset variables
        # set the lower and upper bounds accordingly
        lower_bound.assign(tf.ones(batch_size) * self.lower_bound_const)
        const.assign(tf.ones(batch_size) * self.initial_const)
        upper_bound.assign(tf.ones(batch_size) * self.upper_bound_const)

        # the best l2 and image attack
        o_bestl2.assign(1e10 * tf.ones(batch_size))
        o_bestattack.assign(tf.zeros_like(X))

        @tf.function
        def optim_step(X, y_onehot):
            y = tf.argmax(y_onehot, axis=-1)
            with tf.GradientTape() as find_r_tape:
                X_hat = tanh_reparametrize(X_atanh + r, self.boxplus,
                                           self.boxmul)
                logits_hat = self.model(X_hat)
                y_hat = tf.argmax(logits_hat, axis=-1)
                # Part 1: minimize l2 loss
                l2_dist = tf.reduce_sum(tf.square(X_hat - X), axis=(1, 2, 3))
                l2_loss = tf.reduce_sum(l2_dist)
                # Part 2: classification loss
                real = tf.reduce_sum(y_onehot * logits_hat, 1)
                other = tf.reduce_max(
                    (1 - y_onehot) * logits_hat - y_onehot * 10000, 1)
                if self.TARGETED:
                    # if targetted, optimize for making the other class
                    # most likely
                    cls_loss = tf.maximum(0.0, other - real + self.CONFIDENCE)
                else:
                    # if untargeted, optimize for making this class least
                    # likely.
                    cls_loss = tf.maximum(0.0, real - other + self.CONFIDENCE)
                loss = l2_loss + tf.reduce_sum(const * cls_loss)

            # adjust the best result so far
            if self.TARGETED:
                is_mistake = y == y_hat
            else:
                is_mistake = y != y_hat

            is_best_curr_attack = tf.logical_and(is_mistake, l2_dist < bestl2)
            bestl2.scatter_update(
                to_indexed_slices(l2_dist, batch_indices, is_best_curr_attack))
            bestscore.scatter_update(
                to_indexed_slices(
                    tf.cast(tf.argmax(logits_hat, axis=-1), tf.float32),
                    batch_indices, is_best_curr_attack))
            is_best_attack = tf.logical_and(is_mistake, l2_dist < o_bestl2)
            o_bestl2.scatter_update(
                to_indexed_slices(l2_dist, batch_indices, is_best_attack))
            o_bestattack.scatter_update(
                to_indexed_slices(X_hat, batch_indices, is_best_attack))

            # get gradients and update perturbation
            r_gradients = find_r_tape.gradient(loss, r)
            optimizer.apply_gradients([(r_gradients, r)])

            return loss

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # reset r and optimizer
            r.assign(r0(r.get_shape()))
            [var.assign(tf.zeros_like(var)) for var in optimizer.variables()]

            # reset best l2 and score for inner loop
            bestl2.assign(1e10 * tf.ones(batch_size))
            bestscore.assign(-1 * tf.ones(batch_size))

            prev = 1e10
            for iteration in range(1, self.MAX_ITERATIONS + 1):
                loss = optim_step(X, y_onehot)
                # check if we should abort search if we're getting nowhere.
                if (self.ABORT_EARLY
                        and iteration % (self.MAX_ITERATIONS // 10) == 0):
                    if loss > prev * .9999:
                        break
                    prev = loss

            have_found_solution = tf.not_equal(bestscore, -1)
            # update upper bound
            upper_bound.scatter_update(
                to_indexed_slices(tf.minimum(upper_bound, const),
                                  batch_indices, have_found_solution))
            # update lower bound
            lower_bound.scatter_update(
                to_indexed_slices(tf.maximum(lower_bound, const),
                                  batch_indices, ~have_found_solution))
            # update const (binary search)
            const.assign((lower_bound + upper_bound) / 2)

        return o_bestattack.read_value()

    def call(self, X, y_onehot):
        X_hat = tf.py_function(self._call, [X, y_onehot], tf.float32)
        X_hat.set_shape(X.get_shape())
        return X_hat


class CWL2Prob(object):
    BINARY_SEARCH_STEPS = 9 # number of times to adjust the constant with binary search
    MAX_ITERATIONS = 10000  # number of iterations to perform gradient descent
    ABORT_EARLY = True      # if we stop improving, abort gradient descent early
    LEARNING_RATE = 1e-2    # larger values converge faster to less accurate results
    TARGETED = True         # should we target one specific class? or just be wrong?
    CONFIDENCE = 0.8        # how strong the adversarial example should be
    INITIAL_CONST = 1e-2    # the initial constant c to pick as a first guess

    def __init__(self,
                 model,
                 batch_size=1,
                 prob_confidence=CONFIDENCE,
                 targeted=TARGETED,
                 learning_rate=LEARNING_RATE,
                 binary_search_steps=BINARY_SEARCH_STEPS,
                 max_iterations=MAX_ITERATIONS,
                 abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST,
                 lower_bound=0,
                 upper_bound=1e6,
                 boxmin=0.0,
                 boxmax=1.0):
        """The L_2 optimization attack.
        See CWL2
        """
        super(CWL2Prob, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        self.prob_confidence = prob_confidence
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.initial_const = initial_const
        self.lower_bound_const = lower_bound
        self.upper_bound_const = upper_bound

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmul = (boxmax - boxmin) / 2.
        self.boxplus = (boxmin + boxmax) / 2.

        self.built = False

    def build(self, inputs_shape):
        X_shape, y_shape = inputs_shape
        batch_size = X_shape[0]
        assert y_shape.ndims == 2
        self.r0 = tf.keras.initializers.get("zeros")
        self.r = tf.Variable(self.r0(X_shape), trainable=True, name="cwl2_r")
        self.margin_confidence = get_upper_bound_for_confidence(
            self.prob_confidence, y_shape[-1])
        self.lower_bound = tf.Variable(tf.zeros(batch_size),
                                       trainable=False,
                                       name="lb")
        self.const = tf.Variable(tf.zeros(batch_size),
                                 trainable=False,
                                 name="C")
        self.upper_bound = tf.Variable(tf.zeros(batch_size),
                                       trainable=False,
                                       name="ub")
        self.bestl2 = tf.Variable(tf.zeros(batch_size),
                                  trainable=False,
                                  name="best_l2")
        self.bestscore = tf.Variable(tf.zeros(batch_size),
                                     trainable=False,
                                     name="best_sc")
        self.o_bestl2 = tf.Variable(tf.zeros(batch_size),
                                    trainable=False,
                                    name="o_best_l2")
        self.o_bestattack = tf.Variable(tf.zeros(X_shape),
                                        trainable=False,
                                        name="o_bset_xhat")
        self.optimizer = tf.keras.optimizers.Adam(self.LEARNING_RATE)
        self.built = True

    def __call__(self, X, y_onehot):
        if not self.built:
            inputs_shapes = list(map(lambda x: x.shape, [X, y_onehot]))
            self.build(inputs_shapes)
        X_hat = self.call(X, y_onehot)
        return X_hat

    def _call(self, X, y_onehot):
        batch_size = self.batch_size
        batch_indices = self.batch_indices
        X_atanh = arctanh_reparametrize(X, self.boxplus, self.boxmul)

        ## get variables
        r = self.r
        r0 = self.r0
        optimizer = self.optimizer
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        const = self.const
        o_bestl2 = self.o_bestl2
        o_bestattack = self.o_bestattack
        bestl2 = self.bestl2
        bestscore = self.bestscore

        ## reset variables
        # set the lower and upper bounds accordingly
        lower_bound.assign(tf.ones(batch_size) * self.lower_bound_const)
        const.assign(tf.ones(batch_size) * self.initial_const)
        upper_bound.assign(tf.ones(batch_size) * self.upper_bound_const)

        # the best l2 and image attack
        o_bestl2.assign(1e10 * tf.ones(batch_size))
        o_bestattack.assign(tf.zeros_like(X))

        @tf.function
        def optim_step(X, y_onehot):
            y = tf.argmax(y_onehot, axis=-1)
            with tf.GradientTape() as find_r_tape:
                X_hat = tanh_reparametrize(X_atanh + r, self.boxplus,
                                           self.boxmul)
                logits_hat = self.model(X_hat)
                y_hat = tf.argmax(logits_hat, axis=-1)
                p_hat = tf.reduce_max(tf.nn.softmax(logits_hat), axis=-1)
                # Part 1: minimize l2 loss
                l2_dist = tf.reduce_sum(tf.square(X_hat - X), axis=(1, 2, 3))
                l2_dist = tf.reduce_sum(l2_dist)
                # Part 2: classification loss
                if self.TARGETED:
                    # compute the probability of the label class versus the maximum other
                    real = tf.reduce_sum(y_onehot * logits_hat, 1)
                    other = tf.reduce_max(
                        (1 - y_onehot) * logits_hat - y_onehot * 10000, 1)
                    cls_loss = tf.maximum(0.0, other - real + self.margin_confidence)
                else:
                    # maximize the probability of the mistake
                    # compute the difference between top1 mistake and top2
                    masked_logits = (1 - y_onehot) * logits_hat - 10000 * y_onehot
                    top1, curr_top1 = tf.nn.top_k(masked_logits, k=1)
                    top1, curr_top1 = top1[:, 0], curr_top1[:, 0]
                    curr_top1_onehot = tf.one_hot(curr_top1, y_onehot.shape[-1])
                    top1_masked_logits = (1 - curr_top1_onehot) * logits_hat - 10000 * curr_top1_onehot
                    top2 = tf.reduce_max(top1_masked_logits, axis=1)
                    cls_loss = tf.maximum(0.0, top2 - top1 + self.margin_confidence)
                loss = l2_dist + tf.reduce_sum(const * cls_loss)

            # adjust the best result so far
            if self.TARGETED:
                is_high_confidence_mistake = tf.logical_and(y == y_hat, p_hat > self.prob_confidence)
            else:
                is_high_confidence_mistake = tf.logical_and(y != y_hat, p_hat > self.prob_confidence)

            is_best_curr_attack = tf.logical_and(is_high_confidence_mistake, l2_dist < bestl2)
            bestl2.scatter_update(
                to_indexed_slices(l2_dist, batch_indices, is_best_curr_attack))
            bestscore.scatter_update(
                to_indexed_slices(
                    tf.cast(tf.argmax(logits_hat, axis=-1), tf.float32),
                    batch_indices, is_best_curr_attack))
            is_best_attack = tf.logical_and(is_high_confidence_mistake, l2_dist < o_bestl2)
            o_bestl2.scatter_update(
                to_indexed_slices(l2_dist, batch_indices, is_best_attack))
            o_bestattack.scatter_update(
                to_indexed_slices(X_hat, batch_indices, is_best_attack))

            # get gradients and update perturbation
            r_gradients = find_r_tape.gradient(loss, r)
            optimizer.apply_gradients([(r_gradients, r)])

            return loss

        for outer_step in range(self.BINARY_SEARCH_STEPS):
            # reset r and optimizer
            r.assign(r0(r.get_shape()))
            [var.assign(tf.zeros_like(var)) for var in optimizer.variables()]

            # reset best l2 and score for inner loop
            bestl2.assign(1e10 * tf.ones(batch_size))
            bestscore.assign(-1 * tf.ones(batch_size))

            prev = 1e10
            for iteration in range(1, self.MAX_ITERATIONS + 1):
                loss = optim_step(X, y_onehot)
                # check if we should abort search if we're getting nowhere.
                if (self.ABORT_EARLY
                        and iteration % (self.MAX_ITERATIONS // 10) == 0):
                    if loss > prev * .9999:
                        break
                    prev = loss

            have_found_solution = tf.not_equal(bestscore, -1)
            # update upper bound
            upper_bound.scatter_update(
                to_indexed_slices(tf.minimum(upper_bound, const),
                                  batch_indices, have_found_solution))
            # update lower bound
            lower_bound.scatter_update(
                to_indexed_slices(tf.maximum(lower_bound, const),
                                  batch_indices, ~have_found_solution))
            # update const (binary search)
            const.assign((lower_bound + upper_bound) / 2)

        return o_bestattack.read_value()

    def call(self, X, y_onehot):
        X_hat = tf.py_function(self._call, [X, y_onehot], tf.float32)
        X_hat.set_shape(X.get_shape())
        return X_hat


class OptimizerLi(object):
    def __init__(self,
               model,
                 epsilon=0.3,
                 batch_size=1,
                 r0_init='normal',
                 confidence=0.0,
                 targeted=False,
                 learning_rate=5e-2,
                 lambda_learning_rate=5e-2,
                 max_iterations=10000,
                 min_iterations=100,
                 rtol=1e-2,
                 min_l2_perturbation=0.0,
                 max_l2_perturbation=5.0,
                 initial_const=1.0,
                 use_proxy_constraint=False,
                 boxmin=0.0,
                 boxmax=1.0):
        """The L_inf optimization attack.
        """
        super(OptimizerLi, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        self.r0_init = r0_init
        self.confidence = confidence
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.lambda_learning_rate = lambda_learning_rate
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.rtol = rtol
        self.initial_const = initial_const
        self.use_proxy_constraint = use_proxy_constraint
        self.ur = tfd.Uniform(low=min_l2_perturbation, high=max_l2_perturbation)

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmin = boxmin
        self.boxmax = boxmax

        self.built = False

    def build(self, inputs_shape):
        X_shape, y_shape = inputs_shape
        batch_size = X_shape[0]
        assert y_shape.ndims == 2
        self.r0 = tf.keras.initializers.get(self.r0_init)
        self.r = tf.Variable(tf.zeros(X_shape), trainable=True, name="ol2_r")
        self.const = tf.Variable(tf.zeros(batch_size), trainable=True, name="C")
        self.iterations = tf.Variable(tf.zeros(batch_size), trainable=False)
        self.attack = tf.Variable(tf.zeros(X_shape), trainable=False, name="x_hat")
        self.bestli = tf.Variable(tf.zeros(batch_size), trainable=False, name="best_li")
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.constrained_optimizer = tf.keras.optimizers.Adam(
            self.lambda_learning_rate)
        self.built = True

    def _reset(self, X):
        X_shape = X.shape
        batch_size = X_shape[0]
        self.r.assign(self.r0(X_shape))
        self.const.assign(
            tf.ones(batch_size) * tf.math.log(self.initial_const))
        self.iterations.assign(tf.zeros(batch_size))
        self.attack.assign(X)
        self.bestli.assign(1e10 * tf.ones(batch_size))
        [var.assign(tf.zeros_like(var)) for var in self.optimizer.variables()]
        [
            var.assign(tf.zeros_like(var))
            for var in self.constrained_optimizer.variables()
        ]

    def _call(self, X, y_onehot):
        ## get variables
        r = self.r
        batch_size = X.shape[0]
        batch_indices = self.batch_indices

        optimizer = self.optimizer
        constrained_optimizer = self.constrained_optimizer
        const = self.const
        bestli = self.bestli
        attack = self.attack
        iterations = self.iterations

        def margin(logits, y_onehot, targeted=False):
            real = tf.reduce_sum(y_onehot * logits, 1)
            other = tf.reduce_max(
                (1 - y_onehot) * logits - y_onehot * 10000, 1)
            if targeted:
                # if targetted, optimize for making the other class
                # most likely
                cls_con = other - real + self.confidence
            else:
                # if untargeted, optimize for making this class least
                # likely.
                cls_con = real - other + self.confidence
            return cls_con

        @tf.function
        def optim_constrained(X, y_onehot, targeted=False):
            # increment iterations
            iterations.assign_add(tf.ones(batch_size))
            r_v = r.read_value()
            t = 10.0
            with tf.GradientTape(persistent=True) as find_r_tape:
                X_hat = X + r
                logits_hat = self.model(X_hat)
                # Part 1: minimize li loss
                l1_loss = tf.reduce_sum(tf.abs(r), (1, 2, 3))
                l2_loss = tf.reduce_sum(tf.square(r), axis=(1, 2, 3))
                linf_loss = tf.reduce_max(tf.abs(r), (1, 2, 3))
                linf_loss2 = tf.reduce_sum(
                    tf.nn.relu(tf.abs(r) - self.epsilon), (1, 2, 3))
                linf_loss3 = tf.reduce_logsumexp(tf.abs(r) * t, axis=(1, 2, 3)) / t
                # Part 2: classification loss
                cls_con = margin(logits_hat, y_onehot, targeted=targeted)
                loss = linf_loss3 + tf.exp(const) * tf.nn.relu(cls_con)

            # spectral projected gradient
            is_adv = y != tf.argmax(logits_hat, axis=-1)
            fg = find_r_tape.gradient(loss, r)

            with tf.control_dependencies(
                [optimizer.apply_gradients([(fg, r)])]):
                r.assign(tf.clip_by_value(X + r, 0.0, 1.0) - X)

            if self.use_proxy_constraint:
                multipliers_gradients = -cls_con
            else:
                multipliers_gradients = -tf.sign(cls_con)

            with tf.control_dependencies([
                    constrained_optimizer.apply_gradients([
                        (multipliers_gradients, const)
                    ])
            ]):
                const.assign(
                    tf.clip_by_value(const, tf.math.log(1e-6),
                                     tf.math.log(1e+6)))

            is_best_attack = tf.logical_and(is_adv, linf_loss < bestli)
            bestli.scatter_update(
                to_indexed_slices(linf_loss, batch_indices, is_best_attack))
            attack.scatter_update(
                to_indexed_slices(X_hat, batch_indices, is_best_attack))

            # random restart
            is_conv = li_metric(r - r_v) <= self.rtol
            should_restart = tf.logical_or(tf.logical_and(is_conv, is_adv),
                                           iterations > self.min_iterations)
            rs = self.ur.sample(batch_size)
            bestr = attack - X
            # random perturbation at current best perturbation
            r0 = (bestr + tf.reshape(rs * bestli, (-1, 1, 1, 1)) *
                  tf.sign(tf.random.normal(r.shape)))
            r0 = tf.clip_by_value(X + r0, 0.0, 1.0) - X
            r.scatter_update(
                to_indexed_slices(r0, batch_indices, should_restart))
            iterations.scatter_update(
                to_indexed_slices(tf.zeros_like(iterations), batch_indices,
                                  should_restart))

        # reset optimizer and variables
        self._reset(X)
        # only compute perturbation for correctly classified inputs
        y = tf.argmax(y_onehot, axis=-1)
        corr = prediction(self.model(X)) != y
        bestli.scatter_update(
            to_indexed_slices(tf.zeros_like(bestli), batch_indices, corr))

        for iteration in range(1, self.max_iterations + 1):
            optim_constrained(X, y_onehot, targeted=self.targeted)

        return attack.read_value()

    def __call__(self, X, y_onehot):
        if not self.built:
            inputs_shapes = list(map(lambda x: x.shape, [X, y_onehot]))
            self.build(inputs_shapes)
        X_hat = self.call(X, y_onehot)
        return X_hat

    def call(self, X, y_onehot):
        X_hat = tf.py_function(self._call, [X, y_onehot], tf.float32)
        X_hat.set_shape(X.get_shape())
        return X_hat
