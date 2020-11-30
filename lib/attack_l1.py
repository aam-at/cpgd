from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .attack_lp import (ClassConstrainedAttack, NormConstrainedAttack,
                        PrimalDualGradientAttack,
                        ProximalPrimalDualGradientAttack)
from .attack_utils import hard_threshold, proximal_l1
from .tf_utils import l1_metric


class BaseL1Attack(PrimalDualGradientAttack):
    def __init__(self, model, **kwargs):
        super(BaseL1Attack, self).__init__(model=model, **kwargs)
        self.ord = 1

    def lp_metric(self, u, keepdims=False):
        return l1_metric(u, keepdims=keepdims)


class L1Attack(BaseL1Attack):
    # TODO: add hard thresholding before projection to improve performance
    def __init__(self, model, hard_threshold: bool = True, **kwargs):
        super(L1Attack, self).__init__(model=model, **kwargs)
        self.hard_threshold = hard_threshold

    def _primal_optim_step(self, X, y_onehot):
        # gradient descent on primal variables
        self._primal_optim_step(X, y_onehot)
        rx = self._rx
        if self.hard_threshold:
            lr = self.primal_lr
            th = tf.reshape(lr * self.lambd, (-1, 1, 1, 1))
            rx.assign(hard_threshold(rx, th))


class ClassConstrainedL1Attack(ClassConstrainedAttack, L1Attack):
    pass


class NormConstrainedL1Attack(NormConstrainedAttack, L1Attack):
    pass


class ProximalL1Attack(BaseL1Attack, ProximalPrimalDualGradientAttack):
    def proximity_operator(self, u, l):
        return proximal_l1(u, l)


class ClassConstrainedProximalL1Attack(ClassConstrainedAttack,
                                       ProximalL1Attack):
    pass


class NormConstrainedProximalL1Attack(NormConstrainedAttack,
                                      ProximalL1Attack):
    pass
