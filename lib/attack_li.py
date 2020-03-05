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

    def lp_normalize(self, g):
        return li_normalize(g)

    def proximity_operator(self, u, l):
        return proximal_linf(u, l)
