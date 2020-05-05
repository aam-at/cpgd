import tensorflow as tf
from tensorflow.python.keras import backend

from lib.utils import add_default_end_points
from residual_utils import make_bottleneck_layer


class TsiprasCNN(tf.keras.Model):
    LABEL_RANGES = [(151, 268), (281, 285), (30, 32), (33, 37), (80, 100), (365, 382),
                    (389, 397), (118, 121), (300, 319)]

    # Imagenet robust model Tsipras et al
    def __init__(self):
        super(TsiprasCNN, self).__init__()
        self.backbone = ResnetCNN()

    def call(self, inputs, training=True):
        logits = self.backbone(inputs, training=training)
        num_labels = len(TsiprasCNN.LABEL_RANGES)
        return add_default_end_points({'logits': logits[:, :num_labels]})


class ResnetCNN(tf.keras.Model):
    def __init__(self):
        super(ResnetCNN, self).__init__()

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        # define functional computation graph
        with tf.init_scope():
            z = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv0")(x)
            z = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1e-5, name="conv0/bn")(z)
            z = tf.keras.layers.ReLU()(z)
            z = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(z)
            block1 = make_bottleneck_layer(64, 3, use_bias=False, stride=1, name="group0")
            z = block1(z)
            block2 = make_bottleneck_layer(128, 4, use_bias=False, stride=2, name="group1")
            z = block2(z)
            block3 = make_bottleneck_layer(256, 6, use_bias=False, stride=2, name="group2")
            z = block3(z)
            block4 = make_bottleneck_layer(512, 3, use_bias=False, stride=2, name="group3")
            z = block4(z)
            z = tf.keras.layers.GlobalAveragePooling2D()(z)
            logits = tf.keras.layers.Dense(1000)(z)
        self.model = tf.keras.Model(inputs=x, outputs=logits)

    def call(self, inputs, training=True):
        logits = self.model(inputs, training=training)
        return logits
