from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp

from .attack_lp import OptimizerLp
from .attack_utils import project_box, proximal_l2
from .utils import l2_metric

tfd = tfp.distributions


class OptimizerL2(OptimizerLp):
    def __init__(self, model, **kwargs):
        super(OptimizerL2, self).__init__(model=model, **kwargs)
        self.ord = 2

    def lp_metric(self, u, keepdims=False):
        return l2_metric(u, keepdims=keepdims)

    def proximal_step(self, opt, X, g, l):
        r = self.r
        # generalized gradient after proximity and projection operator
        tl = self.primal_lr * l
        pg = (r - project_box(X, proximal_l2(r - self.primal_lr * g, tl),
                              self.boxmin, self.boxmax)) / self.primal_lr

        with tf.control_dependencies([opt.apply_gradients([(pg, r)])]):
            # final projection
            r.assign(project_box(X, r, self.boxmin, self.boxmax))
