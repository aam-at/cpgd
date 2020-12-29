import warnings

import tensorflow as tf
from cleverhans import utils_tf
from cleverhans.attacks.attack import Attack
from cleverhans.compat import softmax_cross_entropy_with_logits


class SparseL0Descent(Attack):
    """
    This class implements a variant of Projected Gradient Descent for the l0-norm
    (Croce and Hein 2019).
    Paper link (Croce and Hein 2019): https://arxiv.org/pdf/1909.05040.pdf
    """
    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        super(SparseL0Descent, self).__init__(model,
                                              sess=sess,
                                              dtypestr=dtypestr,
                                              **kwargs)
        self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                                'clip_max')
        self.structural_kwargs = [
            'nb_iter', 'rand_init', 'clip_grad', 'sanity_checks'
        ]

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)
        assert self.clip_min is not None
        assert self.clip_max is not None

        asserts = []
        # If a data range was specified, check that the input was in that range
        asserts.append(
            utils_tf.assert_greater_equal(x, tf.cast(self.clip_min, x.dtype)))
        asserts.append(
            utils_tf.assert_less_equal(x, tf.cast(self.clip_max, x.dtype)))

        # Initialize loop variables
        lb = self.clip_min - x
        ub = self.clip_max - x
        if self.rand_init:
            eta = tf.random.uniform(tf.shape(input=x), lb, ub)
        else:
            eta = tf.zeros(tf.shape(input=x))

        # Clip eta
        eta = project_l0_box(eta, self.eps, lb, ub)
        adv_x = x + eta
        if self.clip_min is not None or self.clip_max is not None:
            adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        if self.y_target is not None:
            y = self.y_target
            targeted = True
        elif self.y is not None:
            y = self.y
            targeted = False
        else:
            model_preds = self.model.get_probs(x)
            preds_max = tf.reduce_max(input_tensor=model_preds,
                                      axis=1,
                                      keepdims=True)
            y = tf.cast(tf.equal(model_preds, preds_max), dtype=tf.float32)
            y = tf.stop_gradient(y)
            targeted = False
            del model_preds

        y_kwarg = 'y_target' if targeted else 'y'

        for i in tf.range(self.nb_iter):
            labels, _ = self.get_or_guess_labels(adv_x, {y_kwarg: y})
            logits = self.model.get_logits(adv_x)

            adv_x = sparse_l0_descent(adv_x,
                                      logits,
                                      y=labels,
                                      eps=self.eps_iter,
                                      clip_min=self.clip_min,
                                      clip_max=self.clip_max,
                                      clip_grad=self.clip_grad,
                                      targeted=(self.y_target is not None),
                                      sanity_checks=self.sanity_checks)

            # Clipping perturbation eta to the l1-ball
            eta = adv_x - x
            eta = project_l0_box(eta, self.eps, lb, ub)
            adv_x = x + eta
            # Redo the clipping.
            # Subtracting and re-adding eta can add some small numerical error.
            if self.clip_min is not None or self.clip_max is not None:
                adv_x = utils_tf.clip_by_value(adv_x, self.clip_min,
                                               self.clip_max)

        # Asserts run only on CPU.
        # When multi-GPU eval code tries to force all PGD ops onto GPU, this
        # can cause an error.
        common_dtype = tf.float32
        asserts.append(
            utils_tf.assert_less_equal(
                tf.cast(self.eps_iter, dtype=common_dtype),
                tf.cast(self.eps, dtype=common_dtype)))

        if self.sanity_checks:
            with tf.control_dependencies(asserts):
                adv_x = tf.identity(adv_x)

        return adv_x

    def parse_params(self,
                     eps=10.0,
                     eps_iter=1.0,
                     nb_iter=20,
                     y=None,
                     loss_fn=softmax_cross_entropy_with_logits,
                     clip_min=None,
                     clip_max=None,
                     y_target=None,
                     rand_init=False,
                     clip_grad=False,
                     sanity_checks=True,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (optional float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (optional float) step size for each attack iteration
        :param nb_iter: (optional int) Number of attack iterations.
        :param y: (optional) A tensor with the true labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                        y_target=None if y is also set. Labels should be
                        one-hot-encoded.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param clip_grad: (optional bool) Ignore gradient components
                        at positions where the input is already at the boundary
                        of the domain, and the update step will get clipped out.
        :param sanity_checks: bool Insert tf asserts checking values
            (Some tests need to run with no sanity checks because the
            tests intentionally configure the attack strangely)
        """
        # Save attack-specific parameters
        self.eps = eps
        self.rand_init = rand_init
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.loss_fn = loss_fn
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_grad = clip_grad

        if isinstance(eps, float) and isinstance(eps_iter, float):
            # If these are both known at compile time, we can check before anything
            # is run. If they are tf, we can't check them yet.
            assert eps_iter <= eps, (eps_iter, eps)

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")

        if self.clip_grad and (self.clip_min is None or self.clip_max is None):
            raise ValueError(
                "Must set clip_min and clip_max if clip_grad is set")

        self.sanity_checks = sanity_checks

        if len(kwargs.keys()) > 0:
            warnings.warn("kwargs is unused and will be removed on or after "
                          "2019-04-26.")

        return True


def sparse_l0_descent(x,
                      logits,
                      y=None,
                      eps=1.0,
                      loss_fn=softmax_cross_entropy_with_logits,
                      clip_min=None,
                      clip_max=None,
                      clip_grad=False,
                      targeted=False,
                      sanity_checks=True):
    """
    TensorFlow implementation of the L0 Descent Method.
    """

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(
            utils_tf.assert_greater_equal(x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
        asserts.append(
            utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        preds_max = tf.reduce_max(logits, 1, keepdims=True)
        y = tf.cast(tf.equal(logits, preds_max), dtype=tf.float32)
        y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = loss_fn(labels=y, logits=logits)
    if targeted:
        loss = -loss

    # Define gradient of loss wrt input
    grad, = tf.gradients(ys=loss, xs=x)

    if clip_grad:
        grad = utils_tf.zero_out_clipped_grads(grad, x, clip_min, clip_max)

    red_ind = list(range(1, len(grad.get_shape())))
    optimal_perturbation = grad / (
        1e-10 + tf.reduce_sum(tf.abs(grad), axis=red_ind, keepdims=True))
    # Add perturbation to original example to obtain adversarial example
    t = utils_tf.mul(eps, optimal_perturbation)
    adv_x = (x + t)

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        with tf.control_dependencies(asserts):
            adv_x = tf.identity(adv_x)

    return adv_x


def project_l0_box(x, k, lb, ub):
    k = tf.cast(k, tf.int64)
    p1 = tf.reduce_sum(x**2, axis=-1)
    p2 = tf.minimum(tf.minimum(ub - x, x - lb), 0)
    p2 = tf.reduce_sum(p2**2, axis=-1)
    p3 = tf.sort(tf.reshape(p1 - p2, (x.shape[0], -1)))[:, -k]
    x = tf.clip_by_value(x, lb, ub)
    x *= tf.cast(
        tf.expand_dims((p1 - p2) >= tf.reshape(p3, (-1, 1, 1), -1), -1),
        x.dtype)
    return x
