from __future__ import absolute_import, division, print_function

import tensorflow as tf


def fast_entropy_perturbation(model, X, y=None, epsilon=0.1):
    with tf.GradientTape() as tape:
        tape.watch(X)
        logits = model(X)
        p = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(tf.nn.log_softmax(logits) * p, axis=-1)
    fast_grad_perturbation = epsilon * tf.sign(tape.gradient(entropy, X))
    return tf.stop_gradient(fast_grad_perturbation)


def fast_gradient_perturbation(model, X, y=None, epsilon=0.1):
    with tf.GradientTape() as tape:
        tape.watch(X)
        logits = model(X)
        if y is None:
            # avoid label leaking
            y = tf.argmax(logits, axis=1)
        nll = tf.keras.losses.sparse_categorical_crossentropy(
            y, logits, from_logits=True)
    fast_grad_perturbation = epsilon * tf.sign(tape.gradient(nll, X))
    return tf.stop_gradient(fast_grad_perturbation)


def adversarial_training(model, X, y=None, epsilon=0.1):
    fast_grad_perturbation = fast_gradient_perturbation(model,
                                                        X,
                                                        epsilon=epsilon)
    if y is None:
        logits = model(X)
        y = tf.argmax(logits, axis=1)
    logits_adversarial = model(X + fast_grad_perturbation)
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y, logits_adversarial, from_logits=True)
    return loss


class AdversarialTraining(tf.keras.losses.Loss):
    def __init__(self,
                 model,
                 epsilon=0.1,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name=None):
        super(AdversarialTraining, self).__init__(reduction=reduction,
                                                  name=name)
        self.model = model
        self.epsilon = epsilon

    def __call__(self, X, y=None, sample_weight=None):
        losses = self.call(X, y)
        return tf.reduce_mean(losses)

    def call(self, X, y=None):
        return adversarial_training(self.model, X, y, epsilon=self.epsilon)
