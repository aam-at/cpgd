"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from cleverhans.utils_tf import clip_eta, random_lp_vector
from scipy.spatial.distance import pdist, squareform


class LinfBaseAttack:
    ord = np.inf

    def __init__(self,
                 model,
                 eps: float = 0.3,
                 eps_iter: float = 0.05,
                 nb_iter: int = 10,
                 rand_init: bool = True,
                 rand_init_eps: float = 0.3,
                 loss_fn: str = "xent",
                 early_stopping: bool = True):
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
        self.early_stopping = early_stopping
        self.c = {'xent': 1.1, 'cw': 10.0}[loss_fn]

    @tf.function
    def step_grad(self, x_adv, y):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            logits_adv = self.model(x_adv)
            y_onehot = tf.one_hot(y, logits_adv.shape[-1])
            if self.loss_fn == "xent":
                loss = tf.nn.softmax_cross_entropy_with_logits(y_onehot, logits_adv)
            elif self.loss_fn == "cw":
                label_mask = tf.one_hot(y,
                                        10,
                                        on_value=1.0,
                                        off_value=0.0,
                                        dtype=tf.float32)
                correct_logit = tf.reduce_sum(label_mask * logits_adv, axis=1)
                wrong_logits = (1 - label_mask) * logits_adv - label_mask * 1e4
                wrong_logit = tf.reduce_max(wrong_logits, axis=1)
                loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        return logits_adv, tape.gradient(loss, x_adv)


class LinfDGFAttack(LinfBaseAttack):
    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""

        if self.rand_init:
            eta = random_lp_vector(x_nat.shape, self.ord, self.rand_init_eps)
        else:
            eta = tf.zeros_like(x_nat)
        eta = clip_eta(eta, self.ord, self.eps)
        x_adv = tf.clip_by_value(x_nat + eta, 0.0, 1.0)

        for epoch in range(self.nb_iter):
            logits, grad = self.step_grad(x_adv, y)
            kxy, dxkxy = self.wgf_kernel(x_adv)
            if self.early_stopping:
                # stop update since we already find adversarial perturbation
                is_adv = tf.argmax(logits, axis=-1) != y
                x_adv[is_adv] += self.eps_iter * np.sign(self.c * dxkxy +
                                                         grad)[is_adv]
            x_adv = np.clip(x_adv, x_nat - self.eps, x_nat + self.eps)
            x_adv = np.clip(x_adv, 0, 1)  # ensure valid pixel range

        return x_adv

    def wgf_kernel(self, theta):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2

        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        Kxy = np.exp(-pairwise_dists / h**2 / 2)
        Kxy = np.multiply((pairwise_dists / h**2 / 2 - 1), Kxy)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)

        return (Kxy, dxkxy)


class LinfBLOBAttack(LinfBaseAttack):
    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""

        if self.rand_init:
            eta = random_lp_vector(x_nat.shape, self.ord, self.rand_init_eps)
        else:
            eta = tf.zeros_like(x_nat)
        eta = clip_eta(eta, self.ord, self.eps)
        x_adv = tf.clip_by_value(x_nat + eta, 0.0, 1.0)

        batch_size = x_adv.shape[0]
        for epoch in range(self.nb_iter):
            logits, grad = self.step_grad(x_adv, y)
            kxy, dxkxy = self.svgd_kernel(x_adv)
            if self.early_stopping:
                # stop update since we already find adversarial perturbation
                is_adv = tf.argmax(logits, axis=-1) != y
                x_adv[is_adv] += self.eps_iter * np.sign(
                    self.c * (-(np.matmul(kxy, -grad) + dxkxy) / batch_size) +
                    grad)[is_adv]

            x_adv = np.clip(x_adv, x_nat - self.epsilon, x_nat + self.epsilon)
            x_adv = np.clip(x_adv, 0, 1)  # ensure valid pixel range

        return x_adv

    def svgd_kernel(self, theta):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2

        h = np.median(pairwise_dists)
        h = np.sqrt(0.5 * h / np.log(theta.shape[0]))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
