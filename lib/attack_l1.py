from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .attack_utils import random_lp_vector
from .utils import (l1_metric, prediction,
                    project_log_distribution_wrt_kl_divergence,
                    to_indexed_slices)

tfd = tfp.distributions


class OptimizerL1(object):
    def __init__(
        self,
        model,
        batch_size=1,
        # parameters for the optimizer
        learning_rate=1e-2,
        lambda_learning_rate=1e-1,
        max_iterations=10000,
        # parameters for the attack
        confidence=0.0,
        targeted=False,
        multitargeted=False,
        # parameters for random restarts
        tol=1e-3,
        min_iterations_per_start=0,
        max_iterations_per_start=100,
        sampling_radius=None,
        # parameters for non-convex constrained minimization
        initial_const=0.1,
        minimal_const=1e-6,
        use_proxy_constraint=False,
        boxmin=0.0,
        boxmax=1.0):
        """The L_1 optimization attack.
        """
        super(OptimizerL1, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        # parameters for the optimizer
        self.learning_rate = learning_rate
        self.lambda_learning_rate = lambda_learning_rate
        self.max_iterations = max_iterations
        # parameters for the attack
        self.confidence = confidence
        self.targeted = targeted
        if multitargeted:
            assert not targeted
        self.multitargeted = multitargeted
        # parameters for the random restarts
        self.tol = tol
        self.min_iterations_per_start = min_iterations_per_start
        self.max_iterations_per_start = max_iterations_per_start
        if sampling_radius is not None:
            assert sampling_radius >= 0
        self.sampling_radius = sampling_radius
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
        # attack variables
        self.r = tf.Variable(tf.zeros(X_shape), trainable=True, name="ol1_r")
        self.state = tf.Variable(
            tf.zeros((batch_size, 2)),
            trainable=True,
            constraint=project_log_distribution_wrt_kl_divergence,
            name="state")
        self.iterations = tf.Variable(tf.zeros(batch_size), trainable=False)
        self.attack = tf.Variable(tf.zeros(X_shape),
                                  trainable=False,
                                  name="x_hat")
        self.bestl1 = tf.Variable(tf.zeros(batch_size),
                                  trainable=False,
                                  name="best_l1")
        self.built = True

    def _init_r(self, X):
        if self.sampling_radius is not None and self.sampling_radius > 0:
            r0 = random_lp_vector(X.shape, 1, self.sampling_radius)
        else:
            r0 = tf.zeros(X.shape)
        return r0

    def _reset(self, X):
        X_shape = X.shape
        batch_size = X_shape[0]
        initial_one = np.log(1 - self.initial_const)
        initial_zero = np.log(self.initial_const)
        self.r.assign(self._init_r(X))
        self.state.assign(
            np.stack((np.ones(batch_size) * initial_one,
                      np.ones(batch_size) * initial_zero),
                     axis=1))
        self.iterations.assign(tf.zeros(batch_size))
        self.attack.assign(X)
        self.bestl1.assign(1e10 * tf.ones(batch_size))
        [var.assign(tf.zeros_like(var)) for var in self.optimizer.variables()]
        [
            var.assign(tf.zeros_like(var))
            for var in self.constrained_optimizer.variables()
        ]

    def _call(self, X, y_onehot):
        ## get variables
        batch_size = X.shape[0]
        batch_indices = self.batch_indices

        optimizer = self.optimizer
        constrained_optimizer = self.constrained_optimizer
        r = self.r
        state = self.state
        iterations = self.iterations
        attack = self.attack
        bestl1 = self.bestl1

        # indices of the correct predictions
        y = tf.argmax(y_onehot, axis=-1)
        corr = prediction(self.model(X)) != y

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
                # Part 1: minimize li loss
                l1_loss = l1_metric(r)
                # Part 2: classification loss
                cls_con = margin(logits_hat, y_onehot, targeted=targeted)
                state_distr = tf.exp(state)
                loss = (state_distr[:, 0] * l1_loss +
                        state_distr[:, 1] * tf.nn.relu(cls_con))

            # spectral projected gradient
            is_adv = y != tf.argmax(logits_hat, axis=-1)
            fg = find_r_tape.gradient(loss, r)

            with tf.control_dependencies(
                [optimizer.apply_gradients([(fg, r)])]):
                # soft-thresholding operator for L1-lasso
                r.assign(tfp.math.soft_threshold(r, self.tol))
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

            is_best_attack = tf.logical_and(is_adv, l1_loss < bestl1)
            bestl1.scatter_update(
                to_indexed_slices(l1_loss, batch_indices, is_best_attack))
            attack.scatter_update(
                to_indexed_slices(X_hat, batch_indices, is_best_attack))

            # random restart
            if restarts:
                is_conv = l1_metric(r - r_v) <= self.tol
                # stopping condition: run for at least min_restart_iterations
                # if it does not converges
                should_restart = tf.logical_or(
                    tf.logical_and(
                        tf.logical_and(is_conv, is_adv),
                        iterations >= self.min_iterations_per_start),
                    iterations >= self.max_iterations_per_start)
                r0 = self._init_r(X)
                r.scatter_update(
                    to_indexed_slices(r0, batch_indices, should_restart))
                iterations.scatter_update(
                    to_indexed_slices(tf.zeros_like(iterations), batch_indices,
                                      should_restart))

        # reset optimizer and variables
        self._reset(X)
        # only compute perturbation for correctly classified inputs
        bestl1.scatter_update(
            to_indexed_slices(tf.zeros_like(bestl1), batch_indices, corr))
        for iteration in range(1, self.max_iterations + 1):
            optim_constrained(X,
                              y_onehot,
                              targeted=self.targeted,
                              restarts=True)

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
