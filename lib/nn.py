from __future__ import absolute_import, division, print_function

import tensorflow as tf

from .utils import (spectral_norm, spectral_norm_conv2d,
                    spectral_norm_conv2d_transpose)


class SNDense(tf.keras.layers.Dense):
    def __init__(self, units, power_iterations=1, soft_sn=True, **kwargs):
        super(SNDense, self).__init__(units, **kwargs)
        self.power_iterations = power_iterations
        self.soft_sn = soft_sn

    def build(self, inputs_shape):
        super(SNDense, self).build(inputs_shape)
        w = tf.reshape(self.kernel, (-1, self.kernel.shape[-1]))
        w_shape = w.shape
        self.u = self.add_weight(
            shape=(w_shape[0], 1),
            initializer=tf.keras.initializers.RandomNormal(),
            name='u',
            trainable=False)
        self.kernel_org = self.kernel

    def call(self, inputs, training=True):
        self.kernel = spectral_norm(
            self.kernel_org,
            self.u,
            power_iteration_rounds=self.power_iterations,
            soft=self.soft_sn, training=training)
        outputs = super(SNDense, self).call(inputs)
        return outputs


class SNEmbed(tf.keras.layers.Embedding):
    def __init__(self,
                 input_dim,
                 output_dim,
                 power_iterations=1,
                 soft_sn=True,
                 **kwargs):
        super(SNEmbed, self).__init__(input_dim, output_dim, **kwargs)
        self.power_iterations = power_iterations
        self.soft_sn = soft_sn

    def build(self, inputs_shape):
        super(SNEmbed, self).build(inputs_shape)
        embeddings = self.embeddings
        embeddings_shape = embeddings.shape
        self.u = self.add_weight(
            shape=(embeddings_shape[0], 1),
            initializer=tf.keras.initializers.RandomNormal(),
            name='u',
            trainable=False)
        self.embeddings_org = self.embeddings

    def call(self, inputs, training=True):
        self.embeddings = spectral_norm(
            self.embeddings_org,
            self.u,
            power_iteration_rounds=self.power_iterations,
            soft=self.soft_sn, training=training)
        outputs = super(SNEmbed, self).call(inputs)
        return outputs


class SNConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 power_iterations=1,
                 tight_sn=True,
                 soft_sn=True,
                 **kwargs):
        super(SNConv2D, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations
        self.tight_sn = tight_sn
        self.soft_sn = soft_sn

    def build(self, inputs_shape):
        super(SNConv2D, self).build(inputs_shape)
        if self.tight_sn:
            u_shape = [1] + inputs_shape[1:]
            self.u = self.add_weight(
                shape=u_shape,
                initializer=tf.keras.initializers.RandomNormal(),
                name='u',
                trainable=False)
        else:
            w = tf.reshape(self.kernel, (-1, self.kernel.shape[-1]))
            w_shape = w.shape
            self.u = self.add_weight(
                shape=(w_shape[0], 1),
                initializer=tf.keras.initializers.RandomNormal(),
                name='u',
                trainable=False)
        self.kernel_org = self.kernel

    def call(self, inputs, training=True):
        if self.tight_sn:
            self.kernel = spectral_norm_conv2d(
                self.kernel_org,
                self.u,
                strides=self.strides,
                padding=self.padding.upper(),
                power_iteration_rounds=self.power_iterations,
                soft=self.soft_sn,
                data_format="NCHW"
                if self.data_format == "channels_first" else "NHWC",
                training=training)
        else:
            self.kernel = spectral_norm(
                self.kernel_org,
                self.u,
                power_iteration_rounds=self.power_iterations,
                soft=self.soft_sn, training=training)
        outputs = super(SNConv2D, self).call(inputs)
        return outputs


class SNConv2DTranspose(tf.keras.layers.Conv2DTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 power_iterations=1,
                 tight_sn=True,
                 soft_sn=True,
                 **kwargs):
        super(SNConv2DTranspose, self).__init__(filters, kernel_size, **kwargs)
        self.power_iterations = power_iterations
        self.tight_sn = tight_sn
        self.soft_sn = soft_sn

    def build(self, inputs_shape):
        super(SNConv2DTranspose, self).build(inputs_shape)
        if self.tight_sn:
            u_shape = [1] + inputs_shape[1:]
            self.u = self.add_weight(
                shape=u_shape,
                initializer=tf.keras.initializers.RandomNormal(),
                name='u',
                trainable=False)
            self.v_shape = self.compute_output_shape(u_shape)
        else:
            w = tf.reshape(self.kernel, (-1, self.kernel.shape[-1]))
            w_shape = w.shape
            self.u = self.add_weight(
                shape=(w_shape[0], 1),
                initializer=tf.keras.initializers.RandomNormal(),
                name='u',
                trainable=False)
        self.kernel_org = self.kernel

    def call(self, inputs, training=True):
        if self.tight_sn:
            self.kernel = spectral_norm_conv2d_transpose(
                self.kernel_org,
                self.u,
                self.v_shape,
                strides=self.strides,
                padding=self.padding.upper(),
                power_iteration_rounds=self.power_iterations,
                soft=self.soft_sn,
                data_format="NCHW"
                if self.data_format == "channels_first" else "NHWC",
                training=training)
        else:
            self.kernel = spectral_norm(
                self.kernel_org,
                self.u,
                power_iteration_rounds=self.power_iterations,
                soft=self.soft_sn, training=training)
        outputs = super(SNConv2DTranspose, self).call(inputs)
        return outputs
