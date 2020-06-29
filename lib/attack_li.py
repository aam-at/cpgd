from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from .attack_lp import ProximalGradientOptimizerAttack
from .attack_utils import proximal_linf
from .tf_utils import li_metric


class ProximalLiAttack(ProximalGradientOptimizerAttack):
    def __init__(self, model, **kwargs):
        super(ProximalLiAttack, self).__init__(model=model, **kwargs)
        self.ord = np.inf

    def gradient_preprocess(self, g):
        return tf.sign(g)

    def lp_metric(self, u, keepdims=False):
        return li_metric(u, keepdims=keepdims)

    def proximity_operator(self, u, l):
        return proximal_linf(u, l)
