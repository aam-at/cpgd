from __future__ import absolute_import, division, print_function

import functools

import tensorflow as tf

from .attack_lp import (ClassConstrainedAttack, NormConstrainedAttack,
                        ProximalPrimalDualGradientAttack)
from .attack_utils import proximal_l0, proximal_l1, proximal_l12, proximal_l23
from .tf_utils import l0_metric, l0_pixel_metric


class ProximalL0Attack(ProximalPrimalDualGradientAttack):
    def __init__(self,
                 model,
                 operator="l0",
                 pixel_attack=True,
                 channel_dim=-1,
                 **kwargs):
        super(ProximalL0Attack, self).__init__(model=model, **kwargs)
        assert operator in ["l0", "l1/2", "l2/3", "l1"]
        self.operator = operator
        self.pixel_attack = pixel_attack
        self.channel_dim = channel_dim
        self.ord = 0

    def gradient_preprocess(self, g):
        return g

    def lp_metric(self, u, keepdims=False):
        """Compute l0 metric pixelwise (excluding channel dimension)
        """
        if self.pixel_attack:
            return l0_pixel_metric(u,
                                   channel_dim=self.channel_dim,
                                   keepdims=keepdims)
        else:
            return l0_metric(u, keepdims=keepdims)

    def proximity_operator(self, u, l):
        """Compute l0 proximal operator pixelwise (excluding channel dimension)
        """
        operators = {
            "l0": proximal_l0,
            "l1/2": proximal_l12,
            "l2/3": functools.partial(proximal_l23, has_ecc=self.has_ecc),
            "l1": proximal_l1,
        }
        operator = operators[self.operator]
        if self.pixel_attack:
            # apply operator on the largest channel
            # 1) if the largest channel becomes zero, zero the rest of the elements
            # 2) else set the elements to the original values unless it is maximum
            max_indices = tf.argmax(tf.abs(u), axis=self.channel_dim)
            u_c = tf.expand_dims(tf.gather(u, max_indices, batch_dims=3), -1)
            pu_c = operator(u_c, l)
            o = tf.where(tf.abs(pu_c) > 0, u, 0.0)
            out = tf.where(o == u_c, pu_c, o)
            return out
        else:
            return operator(u, l)


class NormConstrainedProximalL0Attack(NormConstrainedAttack, ProximalL0Attack):
    pass


class ClassConstrainedProximalL0Attack(ClassConstrainedAttack,
                                       ProximalL0Attack):
    pass
