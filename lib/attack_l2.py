from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .attack_lp import (ClassConstrainedAttack, NormConstrainedAttack,
                        PrimalDualGradientAttack,
                        ProximalPrimalDualGradientAttack)
from .attack_utils import proximal_l2
from .tf_utils import l2_metric


class L2Attack(PrimalDualGradientAttack):
    def __init__(self, model, **kwargs):
        super(L2Attack, self).__init__(model=model, **kwargs)
        self.ord = 2

    def gradient_preprocess(self, g):
        # TODO: consider l2-norm gradient normalization
        return g

    def lp_metric(self, u, keepdims=False):
        l2_n = l2_metric(u, keepdims=keepdims)
        # NOTE: stop gradient when norm is zero to avoid nans
        return tf.where(l2_n == 0, tf.stop_gradient(l2_n), l2_n)


class ClassConstrainedL2Attack(ClassConstrainedAttack, L2Attack):
    pass


class NormConstrainedL2Attack(NormConstrainedAttack, L2Attack):
    pass


class ProximalL2Attack(L2Attack, ProximalPrimalDualGradientAttack):
    def proximity_operator(self, u, l):
        return proximal_l2(u, l)


class ClassConstrainedProximalL2Attack(ClassConstrainedAttack,
                                       ProximalL2Attack):
    pass


class NormConstrainedProximalL2Attack(NormConstrainedAttack, ProximalL2Attack):
    pass
