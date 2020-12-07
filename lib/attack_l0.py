from __future__ import absolute_import, division, print_function

from .attack_lp import (ClassConstrainedAttack, NormConstrainedAttack,
                        ProximalPrimalDualGradientAttack)
from .attack_utils import proximal_l0, proximal_l1, proximal_l12, proximal_l23
from .tf_utils import l0_pixel_metric


class ProximalL0Attack(ProximalPrimalDualGradientAttack):
    def __init__(self, model, operator="l0", channel_dim=-1, **kwargs):
        super(ProximalL0Attack, self).__init__(model=model, **kwargs)
        assert operator in ["l0", "l1/2", "l2/3", "l1"]
        self.channel_dim = channel_dim
        self.operator = operator
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
        operators = {
            "l0": proximal_l0,
            "l1/2": proximal_l12,
            "l2/3": proximal_l23,
            "l1": proximal_l1,
        }
        return operators[self.operator](u, l)


class NormConstrainedProximalL0Attack(NormConstrainedAttack, ProximalL0Attack):
    pass


class ClassConstrainedProximalL0Attack(ClassConstrainedAttack,
                                       ProximalL0Attack):
    pass
