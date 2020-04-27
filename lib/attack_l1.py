from __future__ import absolute_import, division, print_function

from .attack_lp import ProximalGradientOptimizerAttack
from .attack_utils import proximal_l1
from .utils import l1_metric


class ProximalL1Attack(ProximalGradientOptimizerAttack):
    def __init__(self, model, **kwargs):
        super(ProximalL1Attack, self).__init__(model=model, **kwargs)
        self.ord = 1

    def gradient_preprocess(self, g):
        return g

    def lp_metric(self, u, keepdims=False):
        return l1_metric(u, keepdims=keepdims)

    def proximity_operator(self, u, l):
        return proximal_l1(u, l)
