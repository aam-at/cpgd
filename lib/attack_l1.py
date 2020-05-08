from __future__ import absolute_import, division, print_function

from .attack_lp import GradientOptimizerAttack, ProximalGradientOptimizerAttack
from .attack_utils import proximal_l1
from .utils import l1_metric


class GradientL1Attack(GradientOptimizerAttack):
    # TODO: add hard thresholding before projection to improve performance
    def __init__(self, model, **kwargs):
        super(GradientL1Attack, self).__init__(model=model, **kwargs)
        self.ord = 1

    def gradient_preprocess(self, g):
        return g

    def lp_metric(self, u, keepdims=False):
        return l1_metric(u, keepdims=keepdims)


class ProximalL1Attack(GradientL1Attack, ProximalGradientOptimizerAttack):
    def proximity_operator(self, u, l):
        return proximal_l1(u, l)
