from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .attack_lp import (ClassConstrainedAttack, NormConstrainedAttack,
                        ProximalPrimalDualGradientAttack)
from .attack_utils import proximal_l0, proximal_l1
from .tf_utils import l0_pixel_metric


class ProximalL0Attack(ProximalPrimalDualGradientAttack):
    def __init__(self, model, soft_threshold=False, channel_dim=-1, **kwargs):
        super(ProximalL0Attack, self).__init__(model=model, **kwargs)
        self.channel_dim = channel_dim
        self.soft_threshold = soft_threshold
        self.ord = 0

    def gradient_preprocess(self, g):
        return g

    def lp_metric(self, u, keepdims=False):
        """Compute l0 metric pixelwise (excluding channel dimension)
        """
        return l0_pixel_metric(u,
                               channel_dim=self.channel_dim,
                               keepdims=keepdims)

    def proximity_operator(self, u, l):
        """Compute l0 proximal operator pixelwise (excluding channel dimension)
        """
        if self.soft_threshold:
            u = proximal_l1(u, l)
            return u
        else:
            u_c = tf.reduce_max(tf.abs(u), axis=self.channel_dim, keepdims=True)
            pu_c = proximal_l0(u_c, l)
            return tf.where(tf.abs(pu_c) > 0, u, 0.0)


class NormConstrainedProximalL0Attack(NormConstrainedAttack, ProximalL0Attack):
    pass


class ClassConstrainedProximalL0Attack(ClassConstrainedAttack,
                                       ProximalL0Attack):
    pass
