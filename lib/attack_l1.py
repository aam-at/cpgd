from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp

from .attack_lp import OptimizerLp
from .attack_utils import proximal_l1
from .utils import l1_metric, l1_normalize

tfd = tfp.distributions


class OptimizerL1(OptimizerLp):
    def __init__(self, model, **kwargs):
        super(OptimizerL1, self).__init__(model=model, **kwargs)
        self.ord = 1

    def lp_metric(self, u, keepdims=False):
        return l1_metric(u, keepdims=keepdims)

    def lp_normalize(self, g):
        return l1_normalize(g)

    def proximity_operator(self, u, l):
        return proximal_l1(u, l)

    def proximal_step(self, opt, X, g, lr, lamb):
        r = self.r
        pg = self.proximal_gradient(X, g, lr, lamb)
        with tf.control_dependencies([opt.apply_gradients([(lr * pg, r)])]):
            if self.has_momentum:
                # gradient momentum can change the sparsity:
                # use hard thresholding to restore the perturbation sparsity
                r.assign(tf.where(tf.abs(r) <= lamb, 0.0, r))
            # final projection
            r.assign(self.project_box(X, r))
