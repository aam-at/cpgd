import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications.resnet import ResNet

from lib.utils import add_default_end_points
from .residual_utils import make_bottleneck_layer


class TsiprasCNN(tf.keras.Model):
    # Imagenet robust model Tsipras et al
    def __init__(self):
        super(TsiprasCNN, self).__init__()

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        # define functional computation graph
        with tf.init_scope():
            from tensorflow.python.keras.applications.resnet import stack1
            z = tf.keras.layers.Conv2D(64, 7, stride=2)(x)
            z = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1e-5)(z)
            z = tf.keras.layers.ReLU()(z)
            z = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(z)
            block1 = make_bottleneck_layer(64, 3, use_bias=False)
            z = block1(z)
            block2 = make_bottleneck_layer(128, 4, use_bias=False)
            z = block2(z)
            block3 = make_bottleneck_layer(256, 6, use_bias=False)
            z = block3(z)
            block4 = make_bottleneck_layer(512, 3, use_bias=False)
            z = block4(z)
            z = tf.keras.layers.GlobalAveragePooling2D()(z)
            logits = tf.keras.layers.Dense(1000)
        self.model = tf.keras.Model(inputs=x, outputs=[z, logits])

    def call(self, inputs, training=True):
        z, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': z, 'logits': logits})
