from __future__ import absolute_import, division, print_function

import tensorflow_probability as tfp

from .attack_lp import OptimizerLp
from .attack_utils import proximal_l0
from .utils import l0_metric, l2_normalize

tfd = tfp.distributions


class OptimizerL0(OptimizerLp):
    def __init__(self, model, **kwargs):
        super(OptimizerL0, self).__init__(model=model, **kwargs)
        self.ord = 0

    def lp_metric(self, u, keepdims=False):
        return l0_metric(u, keepdims=keepdims)

    def lp_normalize(self, g):
        return l2_normalize(g)

    def proximity_operator(self, u, l):
        return proximal_l0(u, l)
