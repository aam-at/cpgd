from __future__ import absolute_import, division, print_function

from abc import abstractmethod

import numpy as np
import tensorflow as tf

from .attack_utils import (project_log_distribution_wrt_kl_divergence,
                           random_lp_vector)
from .utils import prediction, to_indexed_slices, l2_normalize


def create_optimizer(opt, lr, **kwargs):
    config = {'learning_rate': lr}
    config.update(kwargs)
    return tf.keras.optimizers.get({'class_name': opt, 'config': config})


def reset_optimizer(opt):
    [var.assign(tf.zeros_like(var)) for var in opt.variables()]


def margin(logits, y_onehot, delta=0.0, targeted=False):
    real = tf.reduce_sum(y_onehot * logits, 1)
    other = tf.reduce_max((1 - y_onehot) * logits - y_onehot * 10000, 1)
    if targeted:
        # if targetted, optimize for making the other class
        # most likely
        cls_con = other - real + delta
    else:
        # if untargeted, optimize for making this class least
        # likely.
        cls_con = real - other + delta
    return cls_con


class OptimizerLp(object):
    """The L_p optimization attack (external regret minimization with
    multiplicative updates).

    """

    def __init__(
        self,
        model,
        batch_size=1,
        # parameters for the optimizer
        gradient_normalize=False,
        optimizer='sgd',
        primal_lr=1e-1,
        finetune=True,
        primal_fn_lr=0.01,
        dual_lr=1e-1,
        max_iterations=10000,
        # parameters for the attack
        confidence=0.0,
        targeted=False,
        # parameters for random restarts
        tol=5e-3,
        min_iterations_per_start=0,
        max_iterations_per_start=100,
        r0_init="uniform",
        sampling_radius=None,
        # parameters for non-convex constrained minimization
        initial_const=0.1,
        minimal_const=1e-6,
        use_proxy_constraint=False,
        boxmin=0.0,
        boxmax=1.0):
        super(OptimizerLp, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        # parameters for the optimizer
        self.optimizer = optimizer
        self.primal_lr = primal_lr
        self.finetune = finetune
        self.primal_fn_lr = primal_fn_lr
        self.dual_lr = dual_lr
        self.max_iterations = max_iterations
        self.gradient_normalize = gradient_normalize
        # parameters for the attack
        self.confidence = confidence
        self.targeted = targeted
        # parameters for the random restarts
        self.tol = tol
        self.min_iterations_per_start = min_iterations_per_start
        self.max_iterations_per_start = max_iterations_per_start
        self.r0_init = r0_init
        if sampling_radius is not None:
            assert sampling_radius >= 0
        self.sampling_radius = sampling_radius
        # parameters for non-convex constrained optimization
        self.initial_const = (minimal_const
                              if initial_const is None else initial_const)
        assert 0 < self.initial_const < 1.0
        self.minimal_const = minimal_const
        self.use_proxy_constraint = use_proxy_constraint
        self.boxmin = boxmin
        self.boxmax = boxmax

        self.built = False

    def build(self, inputs_shape):
        assert not self.built
        X_shape, y_shape = inputs_shape
        batch_size = X_shape[0]
        assert y_shape.ndims == 2
        # primal and dual variable optimizer
        self.primal_opt = create_optimizer(self.optimizer, self.primal_lr)
        self.primal_fn_opt = create_optimizer(self.optimizer,
                                              self.primal_fn_lr)
        self.dual_opt = create_optimizer(self.optimizer, self.dual_lr)
        # create attack variables
        self.r = tf.Variable(tf.zeros(X_shape), trainable=True, name="r")
        self.state = tf.Variable(
            tf.zeros((batch_size, 2)),
            trainable=True,
            constraint=project_log_distribution_wrt_kl_divergence,
            name="dual_state")
        self.iterations = tf.Variable(tf.zeros(batch_size), trainable=False)
        self.attack = tf.Variable(tf.zeros(X_shape),
                                  trainable=False,
                                  name="x_hat")
        self.bestlp = tf.Variable(tf.zeros(batch_size),
                                  trainable=False,
                                  name="best_lp")
        # create optimizer variables
        gs = [(tf.zeros_like(self.r), self.r)]
        self.primal_opt.apply_gradients(gs)
        self.primal_fn_opt.apply_gradients(gs)
        self.dual_opt.apply_gradients(gs)
        self.built = True

    def _init_r0(self, X):
        if self.sampling_radius is not None and self.sampling_radius > 0:
            if self.r0_init == 'sign':
                r0 = tf.sign(tf.random.normal(X.shape))
                r0 = (self.sampling_radius * r0 /
                      self.lp_metric(r0, keepdims=True))
            elif self.r0_init == 'uniform':
                r0 = tf.random.uniform(X.shape, -1.0, 1.0)
                r0 = (self.sampling_radius * r0 /
                      self.lp_metric(r0, keepdims=True))
            elif self.r0_init == 'lp_sphere':
                r0 = random_lp_vector(X.shape, self.ord, self.sampling_radius)
            else:
                raise ValueError
        else:
            r0 = tf.zeros(X.shape)
        r0 = tf.clip_by_value(X + r0, 0.0, 1.0) - X
        return r0

    def _reset_attack(self, X, y):
        X_shape = X.shape
        batch_size = X_shape[0]
        self.attack.assign(X)
        self.r.assign(self._init_r0(X))
        initial_zero = np.log(self.initial_const)
        initial_one = np.log(1 - self.initial_const)
        self.state.assign(
            np.stack((np.ones(batch_size) * initial_zero,
                      np.ones(batch_size) * initial_one),
                     axis=1))
        self.iterations.assign(tf.zeros(batch_size))
        self.bestlp.assign(1e10 * tf.ones(batch_size))
        # indices of the correct predictions
        assert y.ndim == 1
        corr = prediction(self.model(X)) != y
        # only compute perturbation for correctly classified inputs
        self.bestlp.scatter_update(
            to_indexed_slices(tf.zeros_like(self.bestlp), self.batch_indices,
                              corr))
        # reset optimizer
        reset_optimizer(self.primal_opt)
        reset_optimizer(self.primal_fn_opt)
        reset_optimizer(self.dual_opt)

    @abstractmethod
    def lp_metric(self, u, keepdims=False):
        pass

    @abstractmethod
    def proximal_step(self, opt, X, g, l):
        pass

    def _call(self, X, y_onehot):
        # correct prediction
        y = tf.argmax(y_onehot, axis=-1)

        # get variables
        batch_size = self.batch_size
        primal_opt = self.primal_opt
        primal_fn_opt = self.primal_fn_opt
        dual_opt = self.dual_opt
        attack = self.attack
        r = self.r
        state = self.state
        bestlp = self.bestlp
        iterations = self.iterations

        @tf.function
        def optim_step(X,
                       y_onehot,
                       targeted=False,
                       finetune=False,
                       restarts=True):
            r_v = r.read_value()
            # increment iterations
            iterations.assign_add(tf.ones(batch_size))
            with tf.GradientTape(persistent=True) as find_r_tape:
                X_hat = X + r
                logits_hat = self.model(X_hat)
                # Part 1: minimize l1 loss
                lp_loss = self.lp_metric(r)
                # Part 2: classification loss
                cls_con = margin(logits_hat, y_onehot, targeted=targeted)
                loss = tf.nn.relu(cls_con)

            # lambda for proximity operator
            state_distr = tf.exp(state)
            lambd = state_distr[:, 0] / state_distr[:, 1]
            lambd = tf.reshape(lambd, (-1, 1, 1, 1))

            fg = find_r_tape.gradient(loss, r)
            if self.gradient_normalize:
                fg = l2_normalize(fg)
            # generalized gradient (after proximity and projection operator)
            self.proximal_step(primal_fn_opt if finetune else primal_opt, X, fg, lambd)

            if self.use_proxy_constraint:
                multipliers_gradients = -cls_con
            else:
                multipliers_gradients = -tf.sign(cls_con)
            multipliers_gradients = tf.stack(
                (tf.zeros_like(multipliers_gradients), multipliers_gradients),
                axis=1)
            dual_opt.apply_gradients([(multipliers_gradients, state)])

            is_adv = y != tf.argmax(logits_hat, axis=-1)
            is_best_attack = tf.logical_and(is_adv, lp_loss < bestlp)
            bestlp.scatter_update(
                to_indexed_slices(lp_loss, self.batch_indices, is_best_attack))
            attack.scatter_update(
                to_indexed_slices(X_hat, self.batch_indices, is_best_attack))

            # random restart
            if restarts:
                is_conv = self.lp_metric(r - r_v) <= self.tol
                # stopping condition: run for at least min_restart_iterations
                # if it does not converges
                should_restart = tf.logical_or(
                    tf.logical_and(
                        tf.logical_and(is_conv, is_adv),
                        iterations >= self.min_iterations_per_start),
                    iterations >= self.max_iterations_per_start)
                r0 = self._init_r0(X)
                r.scatter_update(
                    to_indexed_slices(r0, self.batch_indices, should_restart))
                iterations.scatter_update(
                    to_indexed_slices(tf.zeros_like(iterations), self.batch_indices,
                                      should_restart))

        # reset optimizer and variables
        self._reset_attack(X, y)
        for iteration in range(1, self.max_iterations + 1):
            optim_step(X, y_onehot, targeted=self.targeted, restarts=True)

        # finetune for 1/10 iterations
        if self.finetune:
            r.assign(attack - X)
            for iteration in range(1, self.max_iterations // 10 + 1):
                optim_step(X,
                           y_onehot,
                           targeted=self.targeted,
                           restarts=False,
                           finetune=True)

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
