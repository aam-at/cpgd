from __future__ import absolute_import, division, print_function

from .attack_lp import GradientOptimizerAttack, ProximalGradientOptimizerAttack
from .attack_utils import proximal_l2
from .utils import l2_metric


class GradientL2Attack(GradientOptimizerAttack):
    def __init__(self, model, **kwargs):
        super(GradientL2Attack, self).__init__(model=model, **kwargs)
        self.ord = 2

    def lp_metric(self, u, keepdims=False):
        return l2_metric(u, keepdims=keepdims)


class ProximalL2Attack(GradientL2Attack, ProximalGradientOptimizerAttack):
    def proximity_operator(self, u, l):
        return proximal_l2(u, l)
