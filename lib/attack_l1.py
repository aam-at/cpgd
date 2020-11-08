from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .attack_lp import (ClassConstrainedGradientOptimizerAttack,
                        GradientOptimizerAttack,
                        NormConstrainedGradientOptimizerAttack,
                        ProximalClassConstrainedGradientOptimizerAttack,
                        ProximalGradientOptimizerAttack,
                        ProximalNormConstrainedGradientOptimizerAttack)
from .attack_utils import hard_threshold, proximal_l1
from .tf_utils import l1_metric


class BaseL1Attack(GradientOptimizerAttack):
    def __init__(self, model, **kwargs):
        super(BaseL1Attack, self).__init__(model=model, **kwargs)
        self.ord = 1

    def lp_metric(self, u, keepdims=False):
        return l1_metric(u, keepdims=keepdims)


class GradientL1Attack(BaseL1Attack):
    # TODO: add hard thresholding before projection to improve performance
    def __init__(self, model, hard_threshold: bool = True, **kwargs):
        super(GradientL1Attack, self).__init__(model=model, **kwargs)
        self.hard_threshold = hard_threshold

    def _primal_optim_step(self, X, y_onehot):
        # gradient descent on primal variables
        self._primal_optim_step(X, y_onehot)
        if self.hard_threshold:
            lr = self.primal_lr
            th = tf.reshape(lr * self.lambd, (-1, 1, 1, 1))
            self.rx.assign(hard_threshold(self.rx, th))


class ClassConstrainedGradientOptimizerL1Attack(ClassConstrainedGradientOptimizerAttack, BaseL1Attack):
    pass


class NormConstrainedGradientOptimizerL1Attack(NormConstrainedGradientOptimizerAttack, BaseL1Attack):
    pass


class ProximalL1Attack(BaseL1Attack, ProximalGradientOptimizerAttack):
    def proximity_operator(self, u, l):
        return proximal_l1(u, l)


class ClassConstrainedProximalGradientOptimizerL1Attack(ProximalClassConstrainedGradientOptimizerAttack,
                                                        ProximalL1Attack):
    pass


class NormConstrainedProximalGradientOptimizerL1Attack(ProximalNormConstrainedGradientOptimizerAttack,
                                                       ProximalL1Attack):
    pass
