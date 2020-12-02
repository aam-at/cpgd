import ast
from contextlib import contextmanager

import numpy as np
import tensorflow as tf

from .tf_utils import create_lr_schedule, l2_metric, random_targets


def margin(logits, y_onehot, delta=0.0, targeted=False):
    real = tf.reduce_sum(y_onehot * logits, 1)
    other = tf.reduce_max((1 - y_onehot) * logits - y_onehot * 10000, 1)
    if targeted:
        # if targetted, optimize for making the other class
        # most likely
        margin = other - real + delta
    else:
        # if untargeted, optimize for making this class least
        # likely.
        margin = real - other + delta
    return margin


def init_r0(shape, epsilon, norm, init='uniform'):
    if epsilon is not None and epsilon > 0:
        if init == 'sign':
            r0 = tf.sign(tf.random.normal(shape))
            r0 = epsilon * r0
        elif init == 'uniform':
            r0 = tf.random.uniform(shape, -1.0, 1.0)
            r0 = epsilon * r0
        elif init == 'lp_sphere':
            r0 = random_lp_vector(shape, norm, epsilon)
        else:
            raise ValueError
    else:
        r0 = tf.zeros(shape)
    return r0


def get_opt_psi(optimizer, var):
    """Return Psi estimate for adaptive proximal gradient. Supported optimizers:
    Adam, AmsGrad
    """
    assert isinstance(optimizer, tf.keras.optimizers.Optimizer)
    iterations = tf.cast(optimizer.iterations, tf.float32)
    if isinstance(optimizer, tf.keras.optimizers.Adam):
        beta_2 = optimizer.beta_2
        v = optimizer.get_slot(var, "v")
        if optimizer.amsgrad:
            psi = tf.sqrt(v)
        else:
            psi = tf.sqrt(v /
                          (1 - tf.pow(beta_2, iterations))) + optimizer.epsilon
    else:
        raise ValueError("Unsupported optimizer %s" % type(optimizer))
    return psi


def hard_threshold(u, th):
    return tf.where(tf.abs(u) <= th, 0.0, u)


def proximal_l0(u, lambd):
    return hard_threshold(u, tf.sqrt(2 * lambd))


def soft_threshold(x, threshold, name=None):
    # taken from tensorflow_probability
    # https://math.stackexchange.com/questions/471339/derivation-of-soft-thresholding-operator
    with tf.name_scope(name or 'soft_threshold'):
        x = tf.convert_to_tensor(x, name='x')
        threshold = tf.convert_to_tensor(threshold,
                                         dtype=x.dtype,
                                         name='threshold')
        return tf.sign(x) * tf.maximum(tf.abs(x) - threshold, 0.)


def proximal_l1(u, lambd):
    return soft_threshold(u, lambd)


def proximal_l2(u, lambd):
    return tf.nn.relu(1 - lambd / l2_metric(u, keepdims=True)) * u


def proximal_linf(u, lambd):
    u_shape = u.shape
    if u.shape.ndims != 2:
        u = tf.reshape(u, (u_shape[0], -1))
    lambd = tf.reshape(lambd, (-1, 1))
    return tf.reshape(u - lambd * project_l1ball(u / lambd, 1.0), u_shape)


def project_simplex(v, z=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert v.shape.ndims == 2
    n, d = v.shape
    u = tf.sort(v, direction="DESCENDING")
    cssv = tf.cumsum(u, axis=-1) - z
    ind = tf.tile(tf.expand_dims(tf.range(d, dtype=tf.float32) + 1, 0), [n, 1])
    cond = tf.where(u - cssv / ind > 0, ind, -1)
    rho = tf.argmax(cond, -1)
    batch_indices = tf.range(n, dtype=rho.dtype)
    rho_idx = tf.stack([batch_indices, rho], axis=1)
    theta = tf.gather_nd(cssv, rho_idx) / tf.cast(rho + 1, tf.float32)
    theta = tf.expand_dims(theta, -1)
    return tf.nn.relu(v - theta)


def project_l1ball(v, radius):
    """Projects values onto the feasible region.
    """
    assert radius > 0
    assert v.shape.ndims == 2
    u = tf.abs(v)

    cond = tf.reduce_sum(u, axis=1) <= radius
    if tf.reduce_all(cond):
        return u

    w = tf.where(tf.reshape(cond, (-1, 1)), v,
                 tf.sign(v) * project_simplex(u, radius))
    return w


def project_log_distribution_wrt_kl_divergence(log_distribution, axis=1):
    """Projects onto the set of log-multinoulli distributions.
    """
    # For numerical reasons, make sure that the largest element is zero before
    # exponentiating.
    log_distribution = log_distribution - tf.reduce_max(
        log_distribution, axis=axis, keepdims=True)
    log_distribution = log_distribution - tf.math.log(
        tf.reduce_sum(tf.exp(log_distribution), axis=axis, keepdims=True))
    return log_distribution


def project_box(x, u, boxmin, boxmax):
    return tf.clip_by_value(x + u, boxmin, boxmax) - x


def random_exponential(shape, rate=1.0, dtype=tf.float32, seed=None):
    """
    Helper function to sample from the exponential distribution, which is not
    included in core TensorFlow.
    """
    return tf.random.gamma(shape,
                           alpha=1,
                           beta=1. / rate,
                           dtype=dtype,
                           seed=seed)


def random_laplace(shape, loc=0.0, scale=1.0, dtype=tf.float32, seed=None):
    """
    Helper function to sample from the Laplace distribution, which is not
    included in core TensorFlow.
    """
    z1 = random_exponential(shape, loc, dtype=dtype, seed=seed)
    z2 = random_exponential(shape, scale, dtype=dtype, seed=seed)
    return z1 - z2


def random_lp_vector(shape, ord, eps, dtype=tf.float32, seed=None):
    """
    Helper function to generate uniformly random vectors from a norm ball of
    radius epsilon.
    :param shape: Output shape of the random sample. The shape is expected to be
                of the form `(n, d1, d2, ..., dn)` where `n` is the number of
                i.i.d. samples that will be drawn from a norm ball of dimension
                `d1*d1*...*dn`.
    :param ord: Order of the norm (mimics Numpy).
              Possible values: np.inf, 1 or 2.
    :param eps: Epsilon, radius of the norm ball.
    """
    if ord not in [np.inf, 1, 2]:
        raise ValueError('ord must be np.inf, 1, or 2.')

    if ord == np.inf:
        r = tf.random.uniform(shape, -eps, eps, dtype=dtype, seed=seed)
    else:

        # For ord=1 and ord=2, we use the generic technique from
        # (Calafiore et al. 1998) to sample uniformly from a norm ball.
        # Paper link (Calafiore et al. 1998):
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=758215&tag=1
        # We first sample from the surface of the norm ball, and then scale by
        # a factor `w^(1/d)` where `w~U[0,1]` is a standard uniform random variable
        # and `d` is the dimension of the ball. In high dimensions, this is roughly
        # equivalent to sampling from the surface of the ball.

        dim = tf.reduce_prod(shape[1:])

        if ord == 1:
            x = random_laplace((shape[0], dim),
                               loc=1.0,
                               scale=1.0,
                               dtype=dtype,
                               seed=seed)
            norm = tf.reduce_sum(tf.abs(x), axis=-1, keepdims=True)
        elif ord == 2:
            x = tf.random.normal((shape[0], dim), dtype=dtype, seed=seed)
            norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))
        else:
            raise ValueError('ord must be np.inf, 1, or 2.')

        w = tf.pow(tf.random.uniform((shape[0], 1), dtype=dtype, seed=seed),
                   1.0 / tf.cast(dim, dtype))
        r = eps * tf.reshape(w * x / norm, shape)

    return r


def build_lr(lr, lr_config=None):
    if lr_config is not None:
        if isinstance(lr_config, str):
            lr_config = ast.literal_eval(lr_config)
        return create_lr_schedule(lr_config['schedule'], **lr_config['config'])
    else:
        return lr


class AttackOptimizationLoop(object):
    def __init__(self,
                 attack,
                 number_restarts: int = 1,
                 r0_sampling_algorithm: str = 'uniform',
                 r0_sampling_epsilon: float = 0.1,
                 r0_ods_init: bool = False,
                 r0_ods_steps: int = 50,
                 c0_initial_const: float = 0.01,
                 multitargeted: bool = False,
                 lr: float = 0.01,
                 lr_config: str = None,
                 dual_lr: float = 0.1,
                 dual_lr_config: str = None,
                 finetune: bool = True,
                 finetune_lr: float = 0.01,
                 finetune_lr_config: str = None,
                 finetune_dual_lr: float = 0.01,
                 finetune_dual_lr_config: str = None):
        assert not (r0_ods_init and multitargeted)
        self.attack = attack
        self.number_restarts = number_restarts
        self.r0_sampling_algorithm = r0_sampling_algorithm
        self.r0_sampling_epsilon = r0_sampling_epsilon
        self.r0_ods_init = r0_ods_init
        self.multitargeted = multitargeted
        self.r0_ods_steps = r0_ods_steps
        self.c0_initial_const = c0_initial_const
        self.lr = build_lr(lr, lr_config)
        self.dual_lr = build_lr(dual_lr, dual_lr_config)
        self.finetune = finetune
        self.finetune_lr = build_lr(finetune_lr, finetune_lr_config)
        self.finetune_dual_lr = build_lr(finetune_dual_lr,
                                         finetune_dual_lr_config)

    def _run_loop(self, X, y_onehot):
        self.attack.restart_attack(X, y_onehot)
        self.attack.primal_lr = self.lr
        self.attack.dual_lr = self.dual_lr
        for i in range(self.number_restarts):
            r0 = init_r0(X.shape, self.r0_sampling_epsilon, self.attack.ord,
                         self.r0_sampling_algorithm)
            r0 = self.attack.project_box(X, r0)
            c0 = self.c0_initial_const
            self.attack.reset_attack(r0, c0)
            # output diversified initialization
            if self.r0_ods_init:
                with ods_init(self.attack, self.r0_ods_steps) as attack:
                    attack.run(X, y_onehot)
                    r0_ods = self.attack.rx.read_value()
                self.attack.reset_attack(r0_ods, c0)
            # multitargeted attack
            if self.multitargeted:
                y_t = random_targets(y_onehot.shape[0], y_onehot)
                y_t_onehot = tf.one_hot(y_t, y_onehot.shape[1])
                self.attack.targeted = True
                self.attack.run(X, y_t_onehot)
            else:
                self.attack.targeted = False
                self.attack.run(X, y_onehot)
        if self.finetune:
            self.attack.primal_lr = self.finetune_lr
            self.attack.dual_lr = self.finetune_dual_lr
            rbest = self.attack.bestsol.read_value() - X
            cbest = self.attack.bestlambd.read_value()
            self.attack.reset_attack(rbest, cbest)
            if self.attack.targeted:
                # run on the same target to finetune
                y_t = tf.argmax(self.attack.model(X + rbest), -1)
                y_t_onehot = tf.one_hot(y_t, y_onehot.shape[1])
                self.attack.run(X, y_t_onehot)
            else:
                self.attack.run(X, y_onehot)

    def run_loop(self, X, y_onehot):
        if tf.executing_eagerly():
            # eager mode
            self._run_loop(X, y_onehot)
            return self.attack.bestsol.read_value()
        else:
            # graph mode
            with tf.control_dependencies(
                [tf.py_function(self._run_loop, [X, y_onehot], [])]):
                return self.attack.bestsol.read_value()


@contextmanager
def ods_init(attack, ods_iterations=50):
    old_iterations = attack.iterations
    old_loss = attack.classification_loss
    old_optim_step = attack.optim_step
    rand_vector = None

    def ods_loss(X, y_onehot):
        nonlocal rand_vector
        logits = attack.model(X)
        if rand_vector is None:
            rand_vector = tf.random.uniform(logits.shape, -1.0, 1.0)
        return tf.reduce_sum(rand_vector * logits, axis=-1)

    @tf.function
    def ods_optim_step(X, y_onehot):
        attack._primal_optim_step(X, y_onehot)

    attack.iterations = ods_iterations
    attack.classification_loss = ods_loss
    attack.optim_step = ods_optim_step
    try:
        yield attack
    finally:
        attack.iterations = old_iterations
        attack.classification_loss = old_loss
        attack.optim_step = old_optim_step
