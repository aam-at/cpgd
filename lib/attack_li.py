from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from .attack_lp import OptimizerLp
from .attack_utils import proximal_linf
from .utils import l2_normalize, li_metric


class OptimizerLi(OptimizerLp):
    def __init__(self, model, use_sign: bool =False, **kwargs):
        super(OptimizerLi, self).__init__(model=model, **kwargs)
        self.use_sign = use_sign
        self.ord = np.inf

    def lp_metric(self, u, keepdims=False):
        return li_metric(u, keepdims=keepdims)

    def lp_normalize(self, g):
        if self.use_sign:
            return tf.sign(g)
        else:
            return l2_normalize(g)

    def proximity_operator(self, u, l):
        return proximal_linf(u, l)
