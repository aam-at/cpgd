from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow_probability as tfp

from .attack_lp import OptimizerLp
from .attack_utils import project_box, proximal_l2
from .utils import l2_metric, l2_normalize

tfd = tfp.distributions


class OptimizerL2(OptimizerLp):
    def __init__(self, model, **kwargs):
        super(OptimizerL2, self).__init__(model=model, **kwargs)
        self.ord = 2

    def lp_metric(self, u, keepdims=False):
        return l2_metric(u, keepdims=keepdims)

    def lp_normalize(self, g):
        return l2_normalize(g)

    def proximity_operator(self, u, l):
        return proximal_l2(u, l)
