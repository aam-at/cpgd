from __future__ import absolute_import, division, print_function

from abc import ABC, abstractmethod

import tensorflow as tf

from .attack_utils import (margin, project_box,
                           project_log_distribution_wrt_kl_divergence)
from .utils import (create_optimizer, prediction, reset_optimizer,
                    to_indexed_slices)


def compute_lambda(state):
    state_distr = tf.exp(state)
    return state_distr[:, 0] / state_distr[:, 1]


def init_state(ratio):
    condition = tf.reduce_all(ratio > 0)
    assert_op = tf.Assert(condition, [ratio])
    with tf.control_dependencies([assert_op]):
        initial_zero = tf.math.log(ratio / (1 + ratio))
        initial_one = tf.math.log(1 / (1 + ratio))
        state = tf.stack((initial_zero, initial_one), axis=1)
    return state


class GradientOptimizerAttack(ABC):
    """The L_p optimization attack (external regret minimization with
    multiplicative updates).

    """
    def __init__(
        self,
        model,
        # parameters for the optimizer
        loss: str = "cw",
        iterations: int = 100,
        primal_optimizer: str = "sgd",
        primal_lr: float = 1e-1,
        gradient_preprocessing: bool = False,
        dual_optimizer: str = "sgd",
        dual_lr: float = 1e-1,
        dual_ema: bool = True,
        # attack parameters
        targeted: bool = False,
        confidence: float = 0.0,
        # parameters for non-convex constrained minimization
        use_proxy_constraint: bool = True,
        boxmin: float = 0.0,
        boxmax: float = 1.0,
        min_dual_ratio: float = 1e-6,
    ):
        """
        Args:
            model: the model function to call which returns logits.
            loss: classification loss to optimize (e.g. margin loss,
            cross-entropy loss).
            iterations: number of the iteration for the attack.
            primal_optimizer: optimizer for the primal variables.
            primal_lr: learning rate for the primal optimizer.
            gradient_preprocessing: if to apply some preprocessing to the gradient (e.g. l2-normalize the gradient).
            dual_optimizer: optimizer for the dual variables.
            dual_lr: learning for the dual optimizer.
            dual_ema: if to use exponential moving average for the dual variables.
            targeted: if the attack is targeted.
            confidence: attack confidence.
            use_proxy_constraint: if to use proxy Lagrangian formulation
            (https://arxiv.org/abs/1804.06500) to update constraints weights
            boxmin: clipping minimum value.
            boxmax: clipping maximum value.
            min_dual_ratio: minimal ratio C after dual state projection

        """
        super(GradientOptimizerAttack, self).__init__()
        self.model = model
        assert loss in ["logit_diff", "cw", "ce"]
        self.loss = loss
        self.iterations = iterations
        self.primal_opt = create_optimizer(primal_optimizer, primal_lr)
        self.gradient_preprocessing = gradient_preprocessing
        self.dual_opt = create_optimizer(dual_optimizer, dual_lr)
        self.dual_ema = dual_ema
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
        self.targeted = targeted
        self.confidence = confidence
        self.use_proxy_constraint = use_proxy_constraint
        self.boxmin = boxmin
        self.boxmax = boxmax
        self.min_dual_ratio = min_dual_ratio
        self.built = False

    @property
    def primal_lr(self):
        if callable(self.primal_opt.lr):
            return self.primal_opt.lr(self.primal_opt.iterations)
        else:
            return self.primal_opt.lr

    @primal_lr.setter
    def primal_lr(self, lr):
        self.primal_opt.lr = lr

    @property
    def dual_lr(self):
        if callable(self.dual_lr):
            return self.dual_opt.lr(self.dual_opt.iterations)
        else:
            return self.dual_opt.lr

    @dual_lr.setter
    def dual_lr(self, lr):
        self.dual_opt.lr = lr

    def build(self, inputs_shape):
        assert not self.built
        X_shape, y_shape = inputs_shape
        batch_size = X_shape[0]
        assert y_shape.ndims == 2
        # primal and dual variable
        self.rx = tf.Variable(tf.zeros(X_shape), trainable=True, name="rx")
        self.state = tf.Variable(
            tf.zeros((batch_size, 2)),
            trainable=True,
            constraint=self.project_state,
            name="dual_state",
        )
        self.lambd_ema = tf.Variable(tf.zeros(batch_size),
                                     trainable=False,
                                     name="lambd_mu")
        self.ema.apply([self.lambd_ema])
        # create optimizer variables
        self.primal_opt.apply_gradients([(tf.zeros_like(self.rx), self.rx)])
        self.dual_opt.apply_gradients([(tf.zeros_like(self.state), self.state)
                                       ])
        # create other attack variables to store the best attack between optimization iteration
        self.attack = tf.Variable(tf.zeros(X_shape),
                                  trainable=False,
                                  name="x_hat")
        self.bestlambd = tf.Variable(tf.zeros(batch_size),
                                     trainable=False,
                                     name="best_lambd")
        self.bestlp = tf.Variable(tf.zeros(batch_size),
                                  trainable=False,
                                  name="best_lp")
        self.built = True

    def project_state(self, u):
        """Projection onto the set of log-multinoulli distributions w.r.t. KL
        divergence.

        Args:
            u: dual state.

        Returns:
            Projected dual state.

        """
        return tf.maximum(
            tf.math.log(self.min_dual_ratio),
            project_log_distribution_wrt_kl_divergence(u),
        )

    def project_box(self, X, u):
        """Projection w.r.t. box constraints.

        Args:
            X: images.
            u: perturbations.

        Returns:
           u such that X + u is within bo constraints.
        """
        return project_box(X, u, self.boxmin, self.boxmax)

    def cls_constraint_and_loss(self, X, y_onehot):
        """Return classification constraints and classification loss.

        Args:
            X: images.
            y_onehot: original labels.
            targeted: if the attack is targeted.

        Returns:
            Classification constraints and classification loss.
        """
        logits = self.model(X)
        cls_constraint = margin(logits, y_onehot, targeted=self.targeted)
        if self.loss == "logit_diff":
            cls_loss = cls_constraint
        elif self.loss == "cw":
            cls_loss = tf.nn.relu(cls_constraint)
        elif self.loss == "ce":
            if self.targeted:
                cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    y_onehot, logits)
            else:
                cls_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
                    y_onehot, logits)
        return cls_constraint, cls_loss

    def total_loss(self, X, y_onehot, r, lambd):
        """Returns total loss (classification loss + perturbation norm constraint loss).
        Args:
            X: original images.
            y_onehot: original labels.
            r: perturbation to the original images X.
            lambd: trade-off between the classification loss and the L_p norm loss.

        Returns:
            Total cost.
        """
        _, cls_loss = self.cls_constraint_and_loss(X + r, y_onehot)
        lp_loss = self.lp_metric(r)
        return cls_loss + lambd * lp_loss

    @property
    def lambd(self):
        return (self.ema.average(self.lambd_ema)
                if self.dual_ema else compute_lambda(self.state))

    @abstractmethod
    def gradient_preprocess(self, g):
        pass

    @abstractmethod
    def lp_metric(self, u, keepdims=False):
        pass

    def _primal_optim_step(self, X, y_onehot):
        # gradient descent on primal variables
        with tf.GradientTape() as find_r_tape:
            X_hat = X + self.rx
            # Part 1: lp loss
            lp_loss = self.lp_metric(self.ry)
            # Part 2: classification loss
            cls_constraint, cls_loss = self.cls_constraint_and_loss(
                X_hat, y_onehot)
            loss = cls_loss + self.lambd * lp_loss

        # compute gradient for primal variables
        fg = find_r_tape.gradient(loss, self.rx)
        if self.gradient_preprocessing:
            fg = self.gradient_preprocess(fg)
        with tf.control_dependencies(
            [self.primal_opt.apply_gradients([(fg, self.rx)])]):
            self.rx.assign(self.project_box(X, self.rx))

    def _dual_optim_step(self, X, y_onehot):
        # gradient ascent on dual variables
        X_hat = X + self.rx
        cls_constraint, _ = self.cls_constraint_and_loss(X_hat, y_onehot)

        if self.use_proxy_constraint:
            constraint_gradients = cls_constraint
        else:
            constraint_gradients = tf.sign(cls_constraint)
        multipliers_gradients = -tf.stack(
            (tf.zeros_like(constraint_gradients), constraint_gradients),
            axis=1)
        self.dual_opt.apply_gradients([(multipliers_gradients, self.state)])

        if self.dual_ema:
            lambd_new = compute_lambda(self.state)
            with tf.control_dependencies([self.lambd_ema.assign(lambd_new)]):
                self.ema.apply([self.lambd_ema])

    def _update_attack_state(self, X, y_onehot):
        # correct prediction
        batch_indices = tf.range(X.shape[0])
        X_hat = X + self.rx
        logits_hat = self.model(X_hat)
        lp_loss = self.lp_metric(self.rx)
        y = tf.argmax(y_onehot, axis=-1)
        # check if it is the best perturbation
        is_mistake = y != tf.argmax(logits_hat, axis=-1)
        is_best_attack = tf.logical_and(is_mistake, lp_loss < self.bestlp)
        self.attack.scatter_update(
            to_indexed_slices(X_hat, batch_indices, is_best_attack))
        self.bestlambd.scatter_update(
            to_indexed_slices(self.lambd, batch_indices, is_best_attack))
        self.bestlp.scatter_update(
            to_indexed_slices(lp_loss, batch_indices, is_best_attack))

    @tf.function
    def restart_attack(self, X, y_onehot):
        """Restart the attack (reset all the attack variables to their initial state).
        Tip: run before attacks on different inputs.

        Args:
            X: original images.
            y_onehot: original labels.

        """
        batch_size = X.shape[0]
        batch_indices = tf.range(batch_size)
        self.attack.assign(X)
        self.bestlambd.assign(1e10 * tf.ones(batch_size))
        self.bestlp.assign(1e10 * tf.ones(batch_size))
        # only compute perturbation for correctly classified inputs
        with tf.control_dependencies([tf.assert_rank(y_onehot, 2)]):
            y = tf.argmax(y_onehot, axis=-1)
            corr = prediction(self.model(X)) != y
        self.bestlp.scatter_update(
            to_indexed_slices(tf.zeros_like(self.bestlp), batch_indices, corr))

    @tf.function
    def reset_attack(self, r0, C0):
        """Reset the attack optimizer and its variables.
        Tip: run before different random restarts for the attack.

        Args:
            r0: initial perturbation.
            C0: initial constant.

        """
        batch_size = r0.shape[0]
        # NOTE: disabling random restart of optimizers sometimes helps because
        # optimizers contains information about the curvature of the surface
        # which may be useful even at different starting point
        # reset primal optimizer and primal variables
        reset_optimizer(self.primal_opt)
        self.rx.assign(r0)
        reset_optimizer(self.dual_opt)
        self.lambd_ema.assign(tf.ones(batch_size) * C0)
        self.ema.average(self.lambd_ema).assign(self.lambd_ema)
        self.state.assign(init_state(self.lambd_ema))

    @tf.function
    def optim_step(self, X, y_onehot):
        # alternating optimization
        # TODO: compare with simultaneous optimization
        self._primal_optim_step(X, y_onehot)
        self._update_attack_state(X, y_onehot)
        self._dual_optim_step(X, y_onehot)

    def __call__(self, X, y_onehot):
        X_hat = self.call(X, y_onehot)
        return X_hat

    def _run(self, X, y_onehot):
        for iteration in range(self.iterations):
            self.optim_step(X, y_onehot)

    def run(self, X, y_onehot):
        tf.py_function(self._run, [X, y_onehot], [])

    def call(self, X, y_onehot):
        self.run(X, y_onehot)
        return self.attack.read_value()


class ProximalGradientOptimizerAttack(GradientOptimizerAttack, ABC):
    def __init__(self,
                 model,
                 accelerated: bool = False,
                 momentum: float = 0.9,
                 adaptive_momentum: bool = False,
                 **kwargs):
        """
        Args:
            accelerated: if to use accelerated proximal gradient https://arxiv.org/pdf/1705.04925.pdf
            momentum: momentum for accelerated proximal gradient
            adaptive_momentum: if to use adaptive momentum accelerated gradient
            **kwargs:
        """
        super(ProximalGradientOptimizerAttack, self).__init__(model=model,
                                                              **kwargs)
        self.accelerated = accelerated
        self.momentum = momentum
        self.adaptive_momentum = adaptive_momentum

    def build(self, inputs_shape):
        super(ProximalGradientOptimizerAttack, self).build(inputs_shape)
        X_shape, _ = inputs_shape
        batch_size = X_shape[0]
        # mirror variable to track momentum for accelerated gradient
        self.ry = tf.Variable(tf.zeros_like(self.rx),
                              trainable=True,
                              name="ry")
        # (adaptive) momentum for accelerated gradient
        self.beta = tf.Variable(tf.zeros(batch_size))

    def _primal_optim_step(self, X, y_onehot):
        # proximal gradient descent on primal variables
        rx, ry = self.rx, self.ry
        batch_indices = tf.range(X.shape[0])
        # proximal gradient descent on primal variables
        with tf.GradientTape() as find_r_tape:
            X_hat = X + ry
            # Part 1: lp loss
            lp_loss = self.lp_metric(ry)
            # Part 2: classification loss
            cls_constraint, cls_loss = self.cls_constraint_and_loss(
                X_hat, y_onehot)

        # select only active indices among all examples in the batch
        update_indxs = batch_indices[cls_constraint > 0]
        # compute gradient for primal variables
        fg = find_r_tape.gradient(cls_loss, ry)
        fg = tf.gather_nd(fg, tf.expand_dims(update_indxs, axis=1))
        if self.gradient_preprocessing:
            fg = self.gradient_preprocess(fg)

        # proximal or accelerated proximal gradient
        lr = self.primal_lr
        rx_v = rx.read_value()
        sparse_fg = tf.IndexedSlices(fg, update_indxs)
        # sparse updates does not work correctly and stil update all the statistics for some optimizer
        # FIXME: consider using LazyAdam from tf.addons
        with tf.control_dependencies(
            [self.primal_opt.apply_gradients([(sparse_fg, ry)])]):
            mu = tf.reshape(lr * self.lambd, (-1, 1, 1, 1))
            rx.assign(self.project_box(X, self.proximity_operator(ry, mu)))
        if self.accelerated:
            rv = self.project_box(
                X, rx + tf.reshape(self.beta, (-1, 1, 1, 1)) * (rx - rx_v))
            F_x = self.total_loss(X, y_onehot, rx, self.lambd)
            F_v = self.total_loss(X, y_onehot, rv, self.lambd)
            ry.assign(tf.where(tf.reshape(F_x <= F_v, (-1, 1, 1, 1)), rx, rv))
            if self.adaptive_momentum:
                self.beta.scatter_mul(
                    tf.IndexedSlices(tf.where(F_x <= F_v, 0.9, 1.0 / 0.9),
                                     batch_indices))
                self.beta.assign(tf.minimum(self.beta, 1.0))
        else:
            ry.assign(rx)

    @tf.function
    def reset_attack(self, r0, C0):
        super(ProximalGradientOptimizerAttack, self).reset_attack(r0, C0)
        self.ry.assign(self.rx)

    @abstractmethod
    def proximity_operator(self, u, l):
        pass
