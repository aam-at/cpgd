from __future__ import absolute_import, division, print_function

from abc import abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from cleverhans.utils_tf import clip_eta, random_lp_vector

from lib.utils import dist_matrix


class LinfBaseAttack:
    ord = np.inf

    def __init__(self,
                 model,
                 eps: float = 0.3,
                 eps_iter: float = 0.05,
                 nb_iter: int = 10,
                 rand_init: bool = True,
                 rand_init_eps: float = 0.3,
                 loss_fn: str = "xent"):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.model = model
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.rand_init = rand_init
        self.rand_init_eps = rand_init_eps
        self.loss_fn = loss_fn
        self.c = {'xent': 1.1, 'cw': 10.0}[loss_fn]

    def loss_grad(self, x_adv, y):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits_adv = self.model(x_adv)
            num_classes = logits_adv.shape[-1]
            y_onehot = tf.one_hot(y, logits_adv.shape[-1])
            if self.loss_fn == "xent":
                loss = tf.nn.softmax_cross_entropy_with_logits(
                    y_onehot, logits_adv)
            elif self.loss_fn == "cw":
                label_mask = tf.one_hot(y,
                                        num_classes,
                                        on_value=1.0,
                                        off_value=0.0,
                                        dtype=tf.float32)
                correct_logit = tf.reduce_sum(label_mask * logits_adv, axis=1)
                wrong_logits = (1 - label_mask) * logits_adv - label_mask * 1e4
                wrong_logit = tf.reduce_max(wrong_logits, axis=1)
                loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        return tape.gradient(loss, x_adv)

    @abstractmethod
    def attack_step(self, x_nat, x_adv, y):
        ...

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""

        if self.rand_init:
            eta = random_lp_vector(x_nat.shape, self.ord, self.rand_init_eps)
        else:
            eta = tf.zeros_like(x_nat)
        eta = clip_eta(eta, self.ord, self.eps)
        x_adv = tf.clip_by_value(x_nat + eta, 0.0, 1.0)

        x_shape = x_nat.shape
        batch_size = x_shape[0]
        x_nat = tf.reshape(x_nat, (batch_size, -1))
        x_adv = tf.reshape(x_adv, (batch_size, -1))
        for epoch in tf.range(self.nb_iter):
            x_adv_iter = self.eps_iter * tf.sign(
                self.attack_step(x_nat, x_adv, y))
            x_adv += x_adv_iter
            x_adv = tf.clip_by_value(x_adv, x_nat - self.eps, x_nat + self.eps)
            x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)

        x_adv = tf.reshape(x_adv, x_shape)
        return x_adv


class LinfDGFAttack(LinfBaseAttack):
    def attack_step(self, x_nat, x_adv, y):
        grad = self.loss_grad(x_adv, y)
        kxy, dxkxy = self.wgf_kernel(x_adv)
        return self.c * dxkxy + grad

    def wgf_kernel(self, theta):
        batch_size = theta.shape[0]

        pairwise_dists = dist_matrix(theta)
        h = tfp.stats.percentile(pairwise_dists,
                                 50.0,
                                 interpolation="midpoint")
        h = tf.sqrt(0.5 * h / tf.math.log(tf.cast(batch_size, tf.float32) + 1))

        Kxy = tf.exp(-pairwise_dists / h**2 / 2)
        Kxy = (pairwise_dists / h**2 / 2 - 1) * Kxy

        dxkxy = -tf.matmul(Kxy, theta)
        sumkxy_tf = tf.reduce_sum(Kxy, axis=1)
        dxkxy += tf.expand_dims(sumkxy_tf, 1) * theta
        return (Kxy, dxkxy)


class LinfBLOBAttack(LinfBaseAttack):
    def attack_step(self, x_nat, x_adv, y):
        batch_size = x_nat.shape[0]
        grad = self.loss_grad(x_adv, y)
        kxy, dxkxy = self.svgd_kernel(x_adv)
        return self.c * (-(tf.matmul(kxy, -grad) + dxkxy) / batch_size) + grad

    def svgd_kernel(self, theta):
        batch_size = theta.shape[0]
        pairwise_dists = dist_matrix(theta)
        h = tfp.stats.percentile(pairwise_dists,
                                 50.0,
                                 interpolation="midpoint")
        h = tf.sqrt(0.5 * h / tf.math.log(tf.cast(batch_size, tf.float32)))
        # compute the rbf kernel
        Kxy = tf.exp(-pairwise_dists / h**2 / 2)
        dxkxy = -tf.matmul(Kxy, theta)
        sumkxy = tf.reduce_sum(Kxy, axis=1)
        dxkxy += tf.expand_dims(sumkxy, 1) * theta
        dxkxy /= h**2
        return (Kxy, dxkxy)
