from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .utils import (li_metric, prediction,
                    project_log_distribution_wrt_kl_divergence,
                    to_indexed_slices)

tfd = tfp.distributions


class OptimizerL2(object):
    def __init__(
        self,
        model,
        batch_size=1,
        # parameters for the optimizer
        learning_rate=5e-2,
        lambda_learning_rate=1e-2,
        max_iterations=10000,
        # parameters for the attack
        r0_init='zeros',
        confidence=0.0,
        targeted=False,
        multitargeted=False,
        # parameters for random restarts
        tol=5e-3,
        min_restart_iterations=10,
        max_restart_iterations=100,
        # parameters for non-convex constrained minimization
        initial_const=None,
        minimal_const=1e-6,
        use_proxy_constraint=False,
        boxmin=0.0,
        boxmax=1.0):
        """The L_2 optimization attack (external regret minimization with multiplicative
        updates).

        """
        super(OptimizerL2, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        # parameters for the optimizer
        self.learning_rate = learning_rate
        self.lambda_learning_rate = lambda_learning_rate
        self.max_iterations = max_iterations
        # parameters for the attack
        self.r0_init = r0_init
        self.confidence = confidence
        self.targeted = targeted
        if multitargeted:
            assert not targeted
        self.multitargeted = multitargeted
        # parameters for the random restarts
        self.tol = tol
        self.min_restart_iterations = min_restart_iterations
        self.max_restart_iterations = max_restart_iterations
        # parameters for non-convex constrained optimization
        self.initial_const = (minimal_const
                              if initial_const is None else initial_const)
        assert 0 < self.initial_const < 1.0
        self.minimal_const = minimal_const
        self.use_proxy_constraint = use_proxy_constraint

        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        self.boxmin = boxmin
        self.boxmax = boxmax

        self.built = False

    def build(self, inputs_shape):
        X_shape, y_shape = inputs_shape
        batch_size = X_shape[0]
        assert y_shape.ndims == 2
        # primal and dual variable optimizer
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.constrained_optimizer = tf.keras.optimizers.Adam(
            self.lambda_learning_rate)
        # attack parameters
        self.r = tf.Variable(tf.zeros(X_shape), trainable=True, name="ol2_r")
        self.state = tf.Variable(
            tf.zeros((batch_size, 2)),
            trainable=True,
            constraint=project_log_distribution_wrt_kl_divergence,
            name="state")
        # attack variables
        self.iterations = tf.Variable(tf.zeros(batch_size), trainable=False)
        self.attack = tf.Variable(tf.zeros(X_shape),
                                  trainable=False,
                                  name="x_hat")
        self.bestl2 = tf.Variable(tf.zeros(batch_size),
                                  trainable=False,
                                  name="best_l2")
        self.built = True

    def _reset(self, X):
        X_shape = X.shape
        batch_size = X_shape[0]
        initial_one = np.log(1 - self.initial_const)
        initial_zero = np.log(self.initial_const)
        self.state.assign(
            np.stack((np.ones(batch_size) * initial_one,
                      np.ones(batch_size) * initial_zero),
                     axis=1))
        self.r.assign(tf.keras.initializers.get(self.r0_init)(X_shape))
        self.iterations.assign(tf.zeros(batch_size))
        self.attack.assign(X)
        self.bestl2.assign(1e10 * tf.ones(batch_size))
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
        num_classes = y_onehot.shape[-1]

        optimizer = self.optimizer
        constrained_optimizer = self.constrained_optimizer
        state = self.state
        bestl2 = self.bestl2
        attack = self.attack
        iterations = self.iterations
        curr_iter = tf.Variable(0, dtype=tf.int64)

        def margin(logits, y_onehot, targeted=False):
            real = tf.reduce_sum(y_onehot * logits, 1)
            other = tf.reduce_max((1 - y_onehot) * logits - y_onehot * 10000,
                                  1)
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
        def optim_constrained(X, y_onehot, targeted=False, restarts=True):
            # increment iterations
            iterations.assign_add(tf.ones(batch_size))
            r_v = r.read_value()
            with tf.GradientTape(persistent=True) as find_r_tape:
                X_hat = X + r
                logits_hat = self.model(X_hat)
                # Part 1: minimize l2 loss
                l2_loss = tf.reduce_sum(tf.square(r), axis=(1, 2, 3))
                # Part 2: classification loss
                cls_con = margin(logits_hat, y_onehot, targeted=targeted)
                state_distr = tf.exp(state)
                loss = (state_distr[:, 0] * l2_loss +
                        state_distr[:, 1] * tf.nn.relu(cls_con))

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
            multipliers_gradients = tf.stack(
                (tf.zeros_like(multipliers_gradients), multipliers_gradients),
                axis=1)
            constrained_optimizer.apply_gradients([(multipliers_gradients,
                                                    state)])

            is_best_attack = tf.logical_and(is_adv, l2_loss < bestl2)
            bestl2.scatter_update(
                to_indexed_slices(l2_loss, batch_indices, is_best_attack))
            attack.scatter_update(
                to_indexed_slices(X_hat, batch_indices, is_best_attack))

            # random restart
            if restarts:
                curr_iter.assign_add(1)
                # from utils import l2_metric
                # is_conv = l2_metric(r - r_v) <= self.tol
                # is_conv = l2_metric(r - r_v) / l2_metric(r) <= self.tol
                is_conv = li_metric(r - r_v) <= self.tol
                should_restart = tf.logical_or(
                    tf.logical_and(tf.logical_and(is_conv, is_adv),
                                   iterations > self.min_restart_iterations),
                    iterations > self.max_restart_iterations)
                bestr = attack - X
                # random perturbation at current best perturbation
                rs = tf.random.uniform(r.shape[1:], maxval=1.0)
                pr = tf.cast(curr_iter / self.max_iterations, tf.float32)
                scale = tf.minimum(tf.pow(1.0 - 1 / tf.sqrt(bestl2), pr),
                                   1.0 - tf.pow(pr, 4))
                rs *= tf.reshape(scale, (-1, 1, 1, 1))
                rs = tf.clip_by_value(rs, 0.20, 1.0)
                r0 = bestr + rs * tf.sign(tf.random.normal(r.shape))
                r0 = tf.clip_by_value(X + r0, 0.0, 1.0) - X
                r.scatter_update(
                    to_indexed_slices(r0, batch_indices, should_restart))
                iterations.scatter_update(
                    to_indexed_slices(tf.zeros_like(iterations), batch_indices,
                                      should_restart))

        if self.multitargeted:
            best_targeted_l2 = []
            best_targeted_attack = []
            logits = self.model(X)
            logits = tf.where(tf.cast(y_onehot, tf.bool),
                              -np.inf * tf.ones_like(logits), logits)
            sorted_targets = tf.argsort(logits,
                                        axis=-1,
                                        direction='DESCENDING')
            for t in tf.split(sorted_targets[:, :-1], num_classes - 1,
                              axis=-1):
                t_onehot = tf.one_hot(tf.reshape(t, (-1, )), num_classes)
                # reset optimizer and variables
                self._reset(X)
                for iteration in range(1, self.max_iterations + 1):
                    optim_constrained(X, t_onehot, targeted=True)
                best_targeted_l2.append(bestl2.read_value())
                best_targeted_attack.append(attack.read_value())

            best_targeted_attack = tf.stack(best_targeted_attack, axis=1)
            best_targeted_l2 = tf.stack(best_targeted_l2, axis=1)
            bestind = tf.argmin(best_targeted_l2, axis=-1)
            return tf.reshape(
                tf.gather(best_targeted_attack,
                          tf.expand_dims(bestind, 1),
                          axis=1,
                          batch_dims=1), X.shape)
        else:
            # reset optimizer and variables
            self._reset(X)
            # only compute perturbation for correctly classified inputs
            y = tf.argmax(y_onehot, axis=-1)
            corr = prediction(self.model(X)) != y
            bestl2.scatter_update(
                to_indexed_slices(tf.zeros_like(bestl2), batch_indices, corr))
            for iteration in range(1, self.max_iterations + 1):
                optim_constrained(X,
                                  y_onehot,
                                  targeted=self.targeted,
                                  restarts=True)

            # final finetuning
            for iteration in range(1, self.max_iterations // 10 + 1):
                optim_constrained(X,
                                  y_onehot,
                                  targeted=self.targeted,
                                  restarts=False)

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
