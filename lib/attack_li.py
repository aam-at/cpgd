from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from .attack_lp import OptimizerLp
from .attack_utils import project_box, proximal_linf
from .utils import li_metric, li_normalize


class OptimizerLi(OptimizerLp):
    def __init__(self, model, **kwargs):
        super(OptimizerLi, self).__init__(model=model, **kwargs)
        self.ord = np.inf

    def lp_metric(self, u, keepdims=False):
        return li_metric(u, keepdims=keepdims)

    def proximal_step(self, opt, X, g, l):
        r = self.r
        # generalized gradient after proximity and projection operator
        tl = self.primal_lr * l
        pg = (r - project_box(X, proximal_linf(r - self.primal_lr * g, tl),
                              self.boxmin, self.boxmax)) / self.primal_lr

        with tf.control_dependencies([opt.apply_gradients([(pg, r)])]):
            # gradient momentum can interfere (project again)
            r.assign(proximal_linf(r, tl))
            # final projection
            r.assign(project_box(X, r, self.boxmin, self.boxmax))
