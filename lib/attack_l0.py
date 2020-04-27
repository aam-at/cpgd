from __future__ import absolute_import, division, print_function

from .attack_lp import ProximalGradientOptimizerAttack
from .attack_utils import proximal_l0
from .utils import l0_metric


class ProximalL0Attack(ProximalGradientOptimizerAttack):
    def __init__(self, model, **kwargs):
        super(ProximalL0Attack, self).__init__(model=model, **kwargs)
        self.ord = 0

    def lp_metric(self, u, keepdims=False):
        return l0_metric(u, keepdims=keepdims)

    def proximity_operator(self, u, l):
        return proximal_l0(u, l)
