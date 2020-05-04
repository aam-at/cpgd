from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .attack_lp import ProximalGradientOptimizerAttack
from .attack_utils import proximal_l0
from .utils import l0_metric


class ProximalL0Attack(ProximalGradientOptimizerAttack):
    def __init__(self, model, channel_dim=-1, **kwargs):
        super(ProximalL0Attack, self).__init__(model=model, **kwargs)
        self.channel_dim = channel_dim
        self.ord = 0

    def gradient_preprocess(self, g):
        return g

    def lp_metric(self, u, keepdims=False):
        """Compute l0 metric pixelwise (excluding channel dimension)
        """
        u_c = tf.reduce_max(tf.abs(u), axis=self.channel_dim)
        return l0_metric(u_c, keepdims=keepdims)

    def proximity_operator(self, u, l):
        """Compute l0 proximal operator pixelwise (excluding channel dimension)
        """
        u_c = tf.reduce_max(tf.abs(u), axis=self.channel_dim, keepdims=True)
        pu_c = proximal_l0(u_c, l)
        return tf.where(tf.abs(pu_c) > 0, u, 0.0)
