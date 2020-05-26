from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .attack_lp import (GradientOptimizerAttack,
                        ProximalGradientOptimizerAttack, compute_lambda)
from .attack_utils import hard_threshold, proximal_l1
from .utils import l1_metric


class BaseL1Attack(GradientOptimizerAttack):
    def __init__(self, model, **kwargs):
        super(BaseL1Attack, self).__init__(model=model, **kwargs)
        self.ord = 1

    def gradient_preprocess(self, g):
        return g

    def lp_metric(self, u, keepdims=False):
        return l1_metric(u, keepdims=keepdims)


class GradientL1Attack(BaseL1Attack):
    # TODO: add hard thresholding before projection to improve performance
    def __init__(self, model, hard_threshold: bool = True, **kwargs):
        super(GradientL1Attack, self).__init__(model=model, **kwargs)
        self.hard_threshold = hard_threshold

    def _primal_optim_step(self, X, y_onehot):
        # gradient descent on primal variables
        with tf.GradientTape() as find_r_tape:
            X_hat = X + self.rx
            # Part 1: lp loss
            lp_loss = self.lp_metric(self.rx)
            # Part 2: classification loss
            cls_constraint, cls_loss = self.cls_constraint_and_loss(
                X_hat, y_onehot)
            loss = cls_loss + self.lambd * lp_loss

        # compute gradient for primal variables
        fg = find_r_tape.gradient(loss, self.rx)
        if self.gradient_preprocessing:
            fg = self.gradient_preprocess(fg)
        with tf.control_dependencies(
                [self.primal_opt.apply_gradients([(fg, self.rx)])]):
            if self.hard_threshold:
                lr = self.primal_lr
                th = tf.reshape(lr * self.lambd, (-1, 1, 1, 1))
                self.rx.assign(
                    self.project_box(X, hard_threshold(self.rx, th)))
            else:
                self.rx.assign(self.project_box(X, self.rx))


class ProximalL1Attack(BaseL1Attack, ProximalGradientOptimizerAttack):
    def proximity_operator(self, u, l):
        return proximal_l1(u, l)
