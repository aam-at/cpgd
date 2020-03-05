import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .utils import l2_metric


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


def proximal_l0(u, lambd):
    return tf.where(u**2 <= lambd, 0, u)


def proximal_l1(u, lambd):
    return tfp.math.soft_threshold(u, lambd)


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
