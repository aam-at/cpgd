from __future__ import absolute_import, division, print_function

import ast
from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf

from .attack_utils import (margin, project_box,
                           project_log_distribution_wrt_kl_divergence)
from .tf_utils import (create_optimizer, prediction, reset_optimizer,
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


class PrimalDualGradientAttack(ABC):
    """Optimization attack (external regret minimization with multiplicative
    updates).

    """
    def __init__(
        self,
        model,
        # parameters for the optimizer
        loss: str = "cw",
        iterations: int = 100,
        primal_opt: str = "sgd",
        primal_lr: float = 1e-1,
        primal_opt_kwargs: Union[dict, str] = "{}",
        gradient_preprocessing: bool = False,
        dual_opt: str = "sgd",
        dual_lr: float = 1e-1,
        dual_ema: bool = True,
        dual_opt_kwargs: Union[dict, str] = "{}",
        simultaneous_updates: bool = False,
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
            primal_opt: optimizer for the primal variables.
            primal_lr: learning rate for the primal optimizer.
            gradient_preprocessing: if to apply some preprocessing to the gradient (e.g. l2-normalize the gradient).
            dual_opt: optimizer for the dual variables.
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
        super(PrimalDualGradientAttack, self).__init__()
        self.model = model
        assert loss in ["cw", "ce"]
        self.loss = loss
        self.iterations = iterations
        if not isinstance(primal_opt_kwargs, dict):
            assert isinstance(primal_opt_kwargs, str)
            primal_opt_kwargs = ast.literal_eval(primal_opt_kwargs)
        self.primal_opt = create_optimizer(primal_opt, primal_lr,
                                           **primal_opt_kwargs)
        self.gradient_preprocessing = gradient_preprocessing
        self.dual_ema = dual_ema
        if not isinstance(dual_opt_kwargs, dict):
            assert isinstance(dual_opt_kwargs, str)
            dual_opt_kwargs = ast.literal_eval(dual_opt_kwargs)
        self.dual_opt = create_optimizer(dual_opt, dual_lr, **dual_opt_kwargs)
        self.simultaneous_updates = simultaneous_updates
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
        if callable(self.dual_opt.lr):
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
        self._rx = tf.Variable(tf.zeros(X_shape), trainable=True, name="rx")
        self._state = tf.Variable(
            tf.zeros((batch_size, 2)),
            trainable=True,
            constraint=self.project_state,
            name="dual_state",
        )
        self._ema = tf.train.ExponentialMovingAverage(decay=0.9)
        self._lambdas_ema = tf.Variable(tf.zeros((batch_size, 2)),
                                        trainable=False,
                                        name="lambdas_mu")
        self._ema.apply([self._lambdas_ema])
        # create optimizer variables
        self.primal_opt.apply_gradients([(tf.zeros_like(self._rx), self._rx)])
        self.dual_opt.apply_gradients([(tf.zeros_like(self._state),
                                        self._state)])
        # create other attack variables to store the best attack between optimization iteration
        self.bestsol = tf.Variable(tf.zeros(X_shape),
                                   trainable=False,
                                   name="x_hat")
        self.bestlambd = tf.Variable(tf.zeros(batch_size),
                                     trainable=False,
                                     name="best_lambd")
        self.bestobj = tf.Variable(tf.zeros(batch_size),
                                   trainable=False,
                                   name="best_obj")
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

    def classification_loss(self, X, y_onehot):
        """Return classification constraints and classification loss.

        Args:
            X: images.
            y_onehot: original or target labels.

        Returns:
            Classification constraints and classification loss.
        """
        logits = self.model(X)
        if self.loss == "cw":
            m = margin(logits, y_onehot, targeted=self.targeted)
            cls_loss = tf.nn.relu(m + 1.0)
        elif self.loss == "ce":
            if self.targeted:
                cls_loss = tf.nn.softmax_cross_entropy_with_logits(
                    y_onehot, logits)
            else:
                cls_loss = -tf.nn.softmax_cross_entropy_with_logits(
                    y_onehot, logits)
        return cls_loss

    @abstractmethod
    def objective(self, X, r, y_onehot):
        pass

    @abstractmethod
    def constraints(self, X, r, y_onehot):
        pass

    def proxy_constraints(self, X, r, y_onehot):
        """Proxy constraints which by default to use original constraints.
        """
        return self.constraints(X, r, y_onehot)

    @abstractmethod
    def state_gradient(self, constraints_gradients):
        pass

    def total_loss(self, X, r, y_onehot):
        """Returns total loss (classification + lp_metric).
        Args:
            X: original images.
            r: perturbation to the original images X.
            y_onehot: original labels.
            lambd: trade-off between the objective and the constraints.

        Returns:
            Total cost.
        """
        objective = self.objective(X, r, y_onehot)
        constraints = self.proxy_constraints(X, r, y_onehot)
        return tf.reduce_sum(self.lambdas * tf.stack(
            (objective, constraints), axis=1),
                             axis=1)

    @property
    def lambdas(self):
        return (self._ema.average(self._lambdas_ema)
                if self.dual_ema else tf.exp(self._state))

    @property
    def lambd(self):
        lambdas = self.lambdas
        return lambdas[:, 0] / lambdas[:, 1]

    def gradient_preprocess(self, g):
        return g

    @abstractmethod
    def lp_metric(self, u, keepdims=False):
        pass

    def _primal_optim_step(self, X, y_onehot):
        # gradient descent on primal variables
        r = self._rx
        with tf.GradientTape() as find_r_tape:
            loss = self.total_loss(X, r, y_onehot)

        # compute gradient for primal variables
        fg = find_r_tape.gradient(loss, r)
        if self.gradient_preprocessing:
            fg = self.gradient_preprocess(fg)
        with tf.control_dependencies(
            [self.primal_opt.apply_gradients([(fg, r)])]):
            r.assign(self.project_box(X, r))

    def _dual_optim_step(self, X, y_onehot):
        # gradient ascent on dual variables
        r = self._rx
        if self.use_proxy_constraint:
            constraint_gradients = self.proxy_constraints(X, r, y_onehot)
        else:
            constraint_gradients = self.constraints(X, r, y_onehot)
        state_gradient = self.state_gradient(constraint_gradients)
        self.dual_opt.apply_gradients([(state_gradient, self._state)])
        # update moving average of dual variables
        if self.dual_ema:
            lambdas_new = tf.exp(self._state)
            with tf.control_dependencies(
                [self._lambdas_ema.assign(lambdas_new)]):
                self._ema.apply([self._lambdas_ema])

    def _update_attack_state(self, X, y_onehot):
        batch_indices = tf.range(X.shape[0])
        r = self._rx
        objective = self.objective(X, r, y_onehot)
        is_feasible = self.constraints(X, r, y_onehot) <= 0
        # check if it is the best perturbation
        is_best_sol = tf.logical_and(objective < self.bestobj, is_feasible)
        self.bestsol.scatter_update(
            to_indexed_slices(X + r, batch_indices, is_best_sol))
        self.bestlambd.scatter_update(
            to_indexed_slices(self.lambd, batch_indices, is_best_sol))
        self.bestobj.scatter_update(
            to_indexed_slices(objective, batch_indices, is_best_sol))

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
        self.bestsol.assign(X)
        self.bestlambd.assign(1e10 * tf.ones(batch_size))
        self.bestobj.assign(1e10 * tf.ones(batch_size))
        # only compute perturbation for correctly classified inputs
        with tf.control_dependencies([tf.assert_rank(y_onehot, 2)]):
            y = tf.argmax(y_onehot, axis=-1)
            corr = prediction(self.model(X)) != y
        self.bestobj.scatter_update(
            to_indexed_slices(tf.zeros_like(self.bestobj), batch_indices,
                              corr))

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
        self._rx.assign(r0)
        reset_optimizer(self.dual_opt)
        self._state.assign(init_state(tf.ones(batch_size) * C0))
        self._lambdas_ema.assign(tf.exp(self._state))
        self._ema.average(self._lambdas_ema).assign(self._lambdas_ema)

    # @tf.function
    def optim_step(self, X, y_onehot):
        # TODO: compare with simultaneous optimization
        r = self._rx
        r_v_0 = r.read_value()
        self._primal_optim_step(X, y_onehot)
        r_v_1 = r.read_value()
        if self.simultaneous_updates:
            r.assign(r_v_0)
            self._dual_optim_step(X, y_onehot)
            r.assign(r_v_1)
        else:
            self._dual_optim_step(X, y_onehot)
        self._update_attack_state(X, y_onehot)

    def __call__(self, X, y_onehot):
        X_hat = self.call(X, y_onehot)
        return X_hat

    def _run(self, X, y_onehot):
        for iteration in range(self.iterations):
            self.optim_step(X, y_onehot)

    def run(self, X, y_onehot):
        if tf.executing_eagerly():
            self._run(X, y_onehot)
        else:
            # graph mode
            tf.py_function(self._run, [X, y_onehot], [])

    def call(self, X, y_onehot):
        self.run(X, y_onehot)
        return self.bestsol.read_value()


class ClassConstrainedAttack(PrimalDualGradientAttack):
    def objective(self, X, r, y_onehot):
        return self.lp_metric(r)

    def constraints(self, X, r, y_onehot):
        logits = self.model(X + r)
        m = margin(logits, y_onehot, targeted=self.targeted)
        return tf.sign(m)

    def proxy_constraints(self, X, r, y_onehot):
        return self.classification_loss(X + r, y_onehot)

    def state_gradient(self, constraint_gradients):
        return -tf.stack(
            (tf.zeros_like(constraint_gradients), constraint_gradients),
            axis=1)


class NormConstrainedAttack(PrimalDualGradientAttack):
    def __init__(self, model, epsilon: float = None, **kwargs):
        super(NormConstrainedAttack, self).__init__(model=model, **kwargs)
        assert epsilon is not None and epsilon > 0
        self.epsilon = epsilon

    def objective(self, X, r, y_onehot):
        return self.classification_loss(X + r, y_onehot)

    def constraints(self, X, r, y_onehot):
        return self.lp_metric(r) - self.epsilon

    def state_gradient(self, constraint_gradients):
        return -tf.stack(
            (constraint_gradients, tf.zeros_like(constraint_gradients)),
            axis=1)


class ProximalPrimalDualGradientAttack(PrimalDualGradientAttack, ABC):
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
        super(ProximalPrimalDualGradientAttack, self).__init__(model=model,
                                                               **kwargs)
        self.accelerated = accelerated
        self.momentum = momentum
        self.adaptive_momentum = adaptive_momentum

    def build(self, inputs_shape):
        super(ProximalPrimalDualGradientAttack, self).build(inputs_shape)
        X_shape, _ = inputs_shape
        batch_size = X_shape[0]
        # variable to update so we track momentum for accelerated gradient
        self.ry = tf.Variable(tf.zeros_like(self._rx),
                              trainable=True,
                              name="ry")
        # (adaptive) momentum for accelerated gradient
        self.beta = tf.Variable(tf.zeros(batch_size))

    def _primal_optim_step(self, X, y_onehot):
        # proximal gradient descent on primal variables
        rx, ry = self._rx, self.ry
        batch_indices = tf.range(X.shape[0])
        # proximal gradient descent on primal variables
        with tf.GradientTape() as find_r_tape:
            X_hat = X + rx
            # Part 1: lp loss (unused, updated using proximal gradient step)
            lp_loss = self.lp_metric(rx)
            # Part 2: classification loss
            cls_loss = self.classification_loss(X_hat, y_onehot)

        # select only active indices among all examples in the batch
        # compute gradient for primal variables
        fg = find_r_tape.gradient(cls_loss, rx)
        if self.gradient_preprocessing:
            fg = self.gradient_preprocess(fg)

        # proximal or accelerated proximal gradient
        lr = self.primal_lr
        ry_v = ry.read_value()
        # FIXME: consider using LazyAdam from tf.addons to do sparse updates
        with tf.control_dependencies(
            [self.primal_opt.apply_gradients([(fg, rx)])]):
            mu = tf.reshape(lr * self.lambd, (-1, 1, 1, 1))
            ry.assign(self.project_box(X, self.proximity_operator(rx, mu)))
        if self.accelerated:
            rv = self.project_box(
                X, ry + tf.reshape(self.beta, (-1, 1, 1, 1)) * (ry - ry_v))
            F_y = self.total_loss(X, ry, y_onehot)
            F_v = self.total_loss(X, rv, y_onehot)
            rx.assign(tf.where(tf.reshape(F_y <= F_v, (-1, 1, 1, 1)), ry, rv))
            if self.adaptive_momentum:
                self.beta.scatter_mul(
                    tf.IndexedSlices(tf.where(F_y <= F_v, 0.9, 1.0 / 0.9),
                                     batch_indices))
                self.beta.assign(tf.minimum(self.beta, 1.0))
        else:
            rx.assign(ry)

    @tf.function
    def reset_attack(self, r0, C0):
        super(ProximalPrimalDualGradientAttack, self).reset_attack(r0, C0)
        self.ry.assign(self._rx)

    @abstractmethod
    def proximity_operator(self, u, l):
        pass
