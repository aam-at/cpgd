import tensorflow as tf


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self,
                 filter_num,
                 stride=1,
                 conv_shortcut=True,
                 use_bias=True,
                 activation=tf.nn.relu,
                 name=None):
        super(BottleNeck, self).__init__(name=name)
        assert callable(activation)
        self.act = activation
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="conv1",
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1/bn")
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=use_bias,
            name="conv2",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="conv2/bn")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filter_num * 4,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="conv3",
        )
        self.bn3 = tf.keras.layers.BatchNormalization(name="conv3/bn")

        if conv_shortcut:
            self.downsample = tf.keras.Sequential(name="shortcut")
            self.downsample.add(
                tf.keras.layers.Conv2D(
                    filters=filter_num * 4,
                    kernel_size=(1, 1),
                    strides=stride,
                    use_bias=use_bias,
                    name="conv",
                ))
            self.downsample.add(tf.keras.layers.BatchNormalization(name="bn"))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = self.act(tf.keras.layers.add([residual, x]))

        return output


def make_bottleneck_layer(filter_num,
                          blocks,
                          stride=1,
                          use_bias=True,
                          activation=tf.nn.relu,
                          name=None):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(
        BottleNeck(
            filter_num,
            stride=stride,
            conv_shortcut=True,
            use_bias=use_bias,
            activation=activation,
            name="block0",
        ))

    for i in range(1, blocks):
        res_block.add(
            BottleNeck(
                filter_num,
                stride=1,
                conv_shortcut=False,
                use_bias=use_bias,
                activation=activation,
                name=f"block{i}",
            ))

    return res_block