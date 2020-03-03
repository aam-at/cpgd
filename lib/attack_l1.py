from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp

from .attack_lp import OptimizerLp
from .attack_utils import project_box, proximal_l1
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

    def proximal_step(self, opt, X, g, l):
        r = self.r
        # generalized gradient after proximity and projection operator
        tl = self.primal_lr * l
        pg = (r - project_box(X, proximal_l1(r - self.primal_lr * g, tl),
                              self.boxmin, self.boxmax)) / self.primal_lr

        with tf.control_dependencies([opt.apply_gradients([(pg, r)])]):
            # gradient momentum can destroy the sparsity of updates
            # use hard thresholding to restore the perturbation sparsity
            r.assign(tf.where(tf.abs(r) <= tl, 0.0, r))
            # final projection
            r.assign(project_box(X, r, self.boxmin, self.boxmax))
