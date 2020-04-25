from __future__ import absolute_import, division, print_function

from abc import abstractmethod

import tensorflow as tf

from .attack_utils import (l2_metric, margin, project_box,
                           project_log_distribution_wrt_kl_divergence,
                           random_lp_vector)
from .utils import (LinearDecay, prediction, random_targets, to_indexed_slices,
                    l2_normalize)


def create_optimizer(opt, lr, **kwargs):
    config = {'learning_rate': lr}
    config.update(kwargs)
    return tf.keras.optimizers.get({'class_name': opt, 'config': config})


def reset_optimizer(opt):
    [var.assign(tf.zeros_like(var)) for var in opt.variables()]


def compute_lambda(state):
    state_distr = tf.exp(state)
    return state_distr[:, 0] / state_distr[:, 1]


class OptimizerLp(object):
    """The L_p optimization attack (external regret minimization with
    multiplicative updates).

    """
    def __init__(
        self,
        model,
        batch_size,
        # parameters for the optimizer
        loss: str = 'cw',
        iterations: int = 100,
        max_iterations: int = 10000,
        primal_optimizer: str = 'sgd',
        gradient_normalize: bool = True,
        accelerated: bool = False,
        adaptive_momentum: bool = False,
        momentum: float = 0.9,
        primal_lr: float = 1e-1,
        primal_min_lr: float = 1e-2,
        lr_decay: bool = False,
        dual_optimizer: str = 'sgd',
        dual_lr: float = 1e-2,
        dual_ema: bool = True,
        finetune: bool = True,
        # attack parameters
        targeted: bool = False,
        confidence: float = 0.0,
        # parameters for random restarts
        r0_init: str = "uniform",
        sampling_radius: float = 0.5,
        # parameters for non-convex constrained minimization
        initial_const: float = 0.1,
        use_proxy_constraint: bool = True,
        boxmin: float = 0.0,
        boxmax: float = 1.0):
        """

        :param model: the function to call which returns logits.
        :param batch_size: batch size.
        :param loss: loss one of 'cw', 'logit_diff', 'ce'
        :param optimizer: optimizer of the primal loss
        :param gradient_normalize: normalize the gradient before computing the update
        :param accelerated: use accelerated proximal gradient descent from https://arxiv.org/pdf/1705.04925.pdf
        :param adaptive_momentum: use adaptive momentum
        :param momentum: momentum for APGnc
        :param iterations: minimal number of iterations before random restart
        :param max_iterations: maximum number of iterations
        :param primal_lr: learning rate for primal variables
        :param primal_min_lr: minimal learning for primal variables for lr decay or for finetuning
        :param dual_lr: learning rate for dual variables
        :param targeted: if the attack is targeted
        :param lr_decay: use learning rate decay
        :param finetune: finetune perturbation with primal_min_lr
        :param confidence: target attack confidence for 'cw' loss
        :param r0_init: random initialization for perturbation
        :param sampling_radius: sampling radius for random initialization
        :param initial_const: initial const for the constraint weight
        :param minimal_const: minimal constraint weight
        :param use_proxy_constraint: use proxy Lagrangian formulation (https://arxiv.org/abs/1804.06500) to update constraints weight
        :param boxmin: box constraint minimum
        :param boxmax: box constraint maximum
        """
        super(OptimizerLp, self).__init__()
        self.model = model
        self.batch_size = batch_size
        self.batch_indices = tf.range(batch_size)
        # parameters for the optimizer
        assert loss in ['logit_diff', 'cw', 'ce']
        self.loss = loss
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.primal_opt = create_optimizer(primal_optimizer, primal_lr)
        self.gradient_normalize = gradient_normalize
        self.accelerated = accelerated
        self.adaptive_momentum = adaptive_momentum
        self.momentum = momentum
        self.primal_lr = primal_lr
        self.primal_min_lr = (primal_min_lr if primal_min_lr is not None else
                              primal_lr / 10.0)
        self.lr_decay = lr_decay
        self.dual_opt = create_optimizer(dual_optimizer, dual_lr)
        self.dual_lr = dual_lr
        self.dual_ema = dual_ema
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9)
        self.finetune = finetune
        # parameters for the attack
        self.targeted = targeted
        self.confidence = confidence
        # parameters for the random restarts
        self.r0_init = r0_init
        if sampling_radius is not None:
            assert sampling_radius >= 0
        self.sampling_radius = sampling_radius
        # parameters for non-convex constrained optimization
        # initial state for dual variable
        assert initial_const is not None
        self.initial_const = initial_const
        # use proxy constraint
        self.use_proxy_constraint = use_proxy_constraint
        # box projection
        self.boxmin = boxmin
        self.boxmax = boxmax

        self.built = False

    def build(self, inputs_shape):
        assert not self.built
        X_shape, y_shape = inputs_shape
        batch_size = self.batch_size
        assert batch_size == X_shape[0]
        assert y_shape.ndims == 2
        # primal and dual variable
        self.rx = tf.Variable(tf.zeros(X_shape), trainable=True, name="rx")
        self.ry = tf.Variable(tf.zeros(X_shape), trainable=True, name="ry")
        self.state = tf.Variable(tf.zeros((batch_size, 2)),
                                 trainable=True,
                                 constraint=self.project_state,
                                 name="dual_state")
        self.lambd_ema = tf.Variable(tf.zeros(batch_size),
                                     trainable=False,
                                     name="lambd_mu")
        self.ema.apply([self.lambd_ema])
        # create optimizer variables
        self.primal_opt.apply_gradients([(tf.zeros_like(self.ry), self.ry)])
        self.dual_opt.apply_gradients([(tf.zeros_like(self.state), self.state)
                                       ])
        # (adaptive) momentum for accelerated gradient
        self.beta = tf.Variable(tf.zeros(batch_size))
        # create other attack variables to store the best attack
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

    @tf.function
    def _init_r0(self, X):
        if self.sampling_radius is not None and self.sampling_radius > 0:
            if self.r0_init == 'sign':
                r0 = tf.sign(tf.random.normal(X.shape))
                r0 = self.sampling_radius * r0
            elif self.r0_init == 'uniform':
                r0 = tf.random.uniform(X.shape, -1.0, 1.0)
                r0 = self.sampling_radius * r0
            elif self.r0_init == 'lp_sphere':
                r0 = random_lp_vector(X.shape, self.ord, self.sampling_radius)
            else:
                raise ValueError
        else:
            r0 = tf.zeros(X.shape)
        return self.project_box(X, r0)

    @tf.function
    def _init_state(self, ratio):
        batch_size = self.batch_size
        condition = tf.reduce_all(ratio > 0)
        assert_op = tf.Assert(condition, [ratio])
        with tf.control_dependencies([assert_op]):
            initial_zero = tf.math.log(ratio / (1 + ratio))
            initial_one = tf.math.log(1 / (1 + ratio))
            state = tf.stack((tf.ones(batch_size) * initial_zero,
                              tf.ones(batch_size) * initial_one),
                             axis=1)
        return state

    @tf.function
    def _reset_attack(self, X, y_onehot):
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
    def _restart_step(self, X, y_onehot):
        batch_size = X.shape[0]
        # NOTE: disabling random restart of optimizers sometimes helps because
        # optimizers contains information about the curvature of the surface
        # which may be useful even at different starting point
        # reset primal optimizer and primal variables
        reset_optimizer(self.primal_opt)
        self.rx.assign(self._init_r0(X))
        self.ry.assign(self.rx)
        # reset dual optimizer and dual variables to best solution
        reset_optimizer(self.dual_opt)
        self.lambd_ema.assign(tf.ones(batch_size) * self.initial_const)
        self.ema.average(self.lambd_ema).assign(self.lambd_ema)
        self.state.assign(self._init_state(self.lambd_ema))
        self.beta.assign(tf.ones(batch_size) * self.momentum)

    @abstractmethod
    def lp_metric(self, u, keepdims=False):
        pass

    @abstractmethod
    def lp_normalize(self, u):
        pass

    def cls_constraint_and_loss(self, X, y_onehot, targeted=False):
        logits = self.model(X)
        cls_constraint = margin(logits, y_onehot, targeted=targeted)
        if self.loss == 'logit_diff':
            cls_loss = cls_constraint
        elif self.loss == 'cw':
            cls_loss = tf.nn.relu(cls_constraint)
        elif self.loss == 'ce':
            if self.targeted:
                cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    y_onehot, logits)
            else:
                cls_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
                    y_onehot, logits)
        return cls_constraint, cls_loss

    def line_search(self, X, y_onehot, r, g, initial_lr, lambd):
        lr = initial_lr * tf.ones((self.batch_size, ))
        loss0 = self.total_loss(X, y_onehot, r)
        for i in tf.range(5):
            lr_ = tf.reshape(lr, (-1, 1, 1, 1))
            pg = self.proximal_gradient(X, g, r, lr_, lambd)
            r_i = r - pg * lr_
            loss_i = self.total_loss(X, y_onehot, r_i)
            rhs = 0.5 / lr * l2_metric(pg)**2
            lhs = loss0 - loss_i
            cond = lhs < rhs
            lr = tf.where(cond, 2.0 * lr, lr)
            if tf.reduce_all(cond):
                break
        return lr

    def total_loss(self, X, y_onehot, r, lambd):
        _, cls_loss = self.cls_constraint_and_loss(X + r,
                                                   y_onehot,
                                                   targeted=self.targeted)
        lp_loss = self.lp_metric(r)
        return cls_loss + lambd * lp_loss

    @abstractmethod
    def proximity_operator(self, u, l):
        pass

    def proximal_gradient(self, X, g, r, lr, lambd):
        # composition of proximity and projection operator
        rnew = self.project_box(
            X, self.proximity_operator(r - lr * g, lr * lambd))
        return (r - rnew) / lr

    def project_state(self, u):
        return tf.maximum(tf.math.log(1e-6),
                          project_log_distribution_wrt_kl_divergence(u))

    def project_box(self, X, u):
        return project_box(X, u, self.boxmin, self.boxmax)

    def _call(self, X, y_onehot):
        batch_indices = self.batch_indices
        # correct prediction
        y = tf.argmax(y_onehot, axis=-1)

        # primal and dual optimizers
        primal_opt = self.primal_opt
        dual_opt = self.dual_opt
        ema = self.ema
        # optimization variables
        rx = self.rx
        ry = self.ry
        state = self.state
        beta = self.beta
        # best solution
        attack = self.attack
        bestlambd = self.bestlambd
        bestlp = self.bestlp
        lambd_ema = self.lambd_ema

        @tf.function
        def optim_step(X, y_onehot, targeted=False):
            # primal optimization step
            with tf.GradientTape() as find_r_tape:
                X_hat = X + ry
                logits_hat = self.model(X_hat)
                # Part 1: lp loss
                lp_loss = self.lp_metric(ry)
                # Part 2: classification loss
                cls_constraint, cls_loss = self.cls_constraint_and_loss(
                    X_hat, y_onehot, targeted=targeted)

            # select only active indices among all examples in the batch
            mask = cls_constraint > 0
            update_indxs = batch_indices[mask]

            # compute gradient for primal variables
            fg = find_r_tape.gradient(cls_loss, ry)
            fg = tf.gather_nd(fg, tf.expand_dims(update_indxs, axis=1))
            if self.gradient_normalize:
                fg = self.lp_normalize(fg)
            lr = primal_opt.lr

            # proximal or accelerated proximal gradient
            rx_v = rx.read_value()
            sparse_fg = tf.IndexedSlices(fg, update_indxs)
            # sparse updates does not work correctly and stil update all the statistics
            # TODO: consider using LazyAdam from tf.addons
            with tf.control_dependencies(
                [primal_opt.apply_gradients([(sparse_fg, ry)])]):
                lambd = (ema.average(lambd_ema)
                         if self.dual_ema else compute_lambda(state))
                mu = tf.reshape(lr * lambd, (-1, 1, 1, 1))
                self.rx.assign(
                    self.project_box(X, self.proximity_operator(ry, mu)))
            if self.accelerated:
                rv = self.project_box(
                    X, rx + tf.reshape(beta, (-1, 1, 1, 1)) * (rx - rx_v))
                F_x = self.total_loss(X, y_onehot, rx, lambd)
                F_v = self.total_loss(X, y_onehot, rv, lambd)
                self.ry.assign(
                    tf.where(tf.reshape(F_x <= F_v, (-1, 1, 1, 1)), rx, rv))
                if self.adaptive_momentum:
                    beta.scatter_mul(
                        tf.IndexedSlices(tf.where(F_x <= F_v, 0.9, 1.0 / 0.9),
                                         batch_indices))
                    beta.assign(tf.minimum(beta, 1.0))
            else:
                ry.assign(rx)

            # dual gradient ascent step (alternating optimization)
            # TODO: compare with simultaneous optimization
            X_hat = X + ry
            logits_hat = self.model(X_hat)
            # Part 1: lp loss
            lp_loss = self.lp_metric(ry)
            # Part 2: classification loss
            cls_constraint, _ = self.cls_constraint_and_loss(X_hat,
                                                             y_onehot,
                                                             targeted=targeted)

            if self.use_proxy_constraint:
                constraint_gradients = cls_constraint
            else:
                constraint_gradients = tf.sign(cls_constraint)
            multipliers_gradients = -tf.stack(
                (tf.zeros_like(lp_loss), constraint_gradients), axis=1)
            dual_opt.apply_gradients([(multipliers_gradients, state)])

            if self.dual_ema:
                lambd_new = compute_lambda(state)
                with tf.control_dependencies([lambd_ema.assign(lambd_new)]):
                    self.ema.apply([lambd_ema])

            # check if it is the best perturbation
            is_mistake = y != tf.argmax(logits_hat, axis=-1)
            is_best_attack = tf.logical_and(is_mistake, lp_loss < bestlp)
            attack.scatter_update(
                to_indexed_slices(X_hat, batch_indices, is_best_attack))
            bestlambd.scatter_update(
                to_indexed_slices(lambd, batch_indices, is_best_attack))
            bestlp.scatter_update(
                to_indexed_slices(lp_loss, batch_indices, is_best_attack))

        # reset optimizer and variables
        self._reset_attack(X, y_onehot)
        primal_opt.lr.assign(self.primal_lr)

        if self.lr_decay:
            lr_decay = LinearDecay(primal_opt.lr, self.primal_min_lr,
                                   self.iterations)
        for iteration in range(self.max_iterations):
            if iteration % self.iterations == 0:
                primal_opt.lr = self.primal_lr
                self._restart_step(X, y_onehot)
            if self.lr_decay:
                # as we decrease learning rate lr * lambd used in proximity
                # operator decreases too.
                # FIXME: adjust the state, so mu remains constant even after
                # learning rate decrease
                new_lr = lr_decay(primal_opt.iterations)
                primal_opt.lr.assign(new_lr)
            optim_step(X, y_onehot, targeted=self.targeted)

        if self.finetune:
            self._restart_step(X, y_onehot)
            # restore best attack to finetune it with smaller learning rate
            rx.assign(attack - X)
            ry.assign(rx)
            state.assign(self._init_state(bestlambd))
            lambd_ema.assign(bestlambd)
            # set smaller learning rate to finetune the attack
            primal_opt.lr.assign(self.primal_min_lr)
            lr_decay = LinearDecay(primal_opt.lr, self.primal_min_lr / 10,
                                   self.iterations)
            # finetune
            for iteration in range(self.iterations):
                new_lr = lr_decay(primal_opt.iterations)
                primal_opt.lr.assign(new_lr)
                optim_step(X, y_onehot, targeted=self.targeted)

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
