import tensorflow as tf
from absl import flags

from lib.utils import add_default_end_points, change_default_args

FLAGS = flags.FLAGS


class TradesCNN(tf.keras.Model):
    def __init__(self, drop=0.5):
        super(TradesCNN, self).__init__()
        self.drop = drop

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        conv = tf.keras.layers.Conv2D
        dense = tf.keras.layers.Dense
        # configure layers
        conv = change_default_args(conv,
                                   data_format='channels_first')
        max_pool = change_default_args(tf.keras.layers.MaxPool2D,
                                       data_format='channels_first')
        dense = change_default_args(dense)
        act = change_default_args(tf.keras.layers.Activation, activation='relu')
        # define functional computation graph
        with tf.init_scope():
            z = conv(32, 3)(x)
            z = act()(z)
            z = conv(32, 3)(z)
            z = act()(z)
            z = max_pool(2)(z)
            z = conv(64, 3)(z)
            z = act()(z)
            z = conv(64, 3)(z)
            z = act()(z)
            z = max_pool(2)(z)
            z = tf.keras.layers.Flatten()(z)
            # classifier
            h = dense(200)(z)
            h = act()(h)
            # h = tf.keras.layers.Dropout(self.drop)(h)
            h = dense(200)(h)
            h = act()(h)
            logits = tf.keras.layers.Dense(10, activation=None)(h)
        self.model = tf.keras.Model(inputs=x, outputs=[z, logits])

    def call(self, inputs, training=True):
        z, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': z, 'logits': logits})


class MadryCNN(tf.keras.Model):
    def __init__(self):
        super(MadryCNN, self).__init__()

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        conv = tf.keras.layers.Conv2D
        dense = tf.keras.layers.Dense
        # configure layers
        conv = change_default_args(conv,
                                   activation='relu',
                                   padding='same',
                                   data_format='channels_last')
        max_pool = change_default_args(tf.keras.layers.MaxPool2D,
                                       data_format='channels_last')
        dense = change_default_args(dense)
        activation = tf.keras.activations.get('relu')
        # define functional computation graph
        with tf.init_scope():
            z = conv(32, 5)(x)
            z = max_pool(2)(z)
            z = conv(64, 5)(z)
            z = max_pool(2)(z)
            z = tf.keras.layers.Flatten()(z)
            h = dense(1024)(z)
            z = tf.keras.layers.Lambda(lambda x: activation(x))(h)
            logits = tf.keras.layers.Dense(10, activation=None)(z)
        self.model = tf.keras.Model(inputs=x, outputs=[h, logits])

    def call(self, inputs, training=True):
        h, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': h, 'logits': logits})
