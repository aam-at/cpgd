import tensorflow as tf
from absl import flags

from lib.utils import add_default_end_points, change_default_args

FLAGS = flags.FLAGS


class MadryCNN(tf.keras.Model):
    # Model trained using adversarial training with projected gradient attack
    def __init__(self, model_type='plain'):
        self.model_type = model_type
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
                                   use_bias=self.model_type == 'plain',
                                   padding='same',
                                   data_format='channels_last')
        dense = change_default_args(dense)
        act = change_default_args(tf.keras.layers.Activation,
                                  activation='relu')
        # define functional computation graph
        with tf.init_scope():
            z = conv(96, 3)(x)
            z = conv(96, 3)(z)
            z = conv(192, 3, strides=2)(z)
            z = conv(192, 3)(z)
            z = conv(192, 3)(z)
            z = conv(192, 3, strides=2)(z)
            z = conv(192, 3)(z)
            z = conv(384, 2, strides=2)(z)
            # classifier
            z = tf.keras.layers.Flatten()(z)
            h = dense(1200)(z)
            z = act()(h)
            logits = dense(10)(z)
        self.model = tf.keras.Model(inputs=x, outputs=[z, logits])

    def call(self, inputs, training=True):
        if self.model_type == 'l2':
            inputs = inputs - tf.ones_like(inputs) * 0.5
        h, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': h, 'logits': logits})
