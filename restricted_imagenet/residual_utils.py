import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, use_bias=True, name=None):
        super(BasicBlock, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=use_bias,
            name="conv1"
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1/bn")
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="conv2"
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="conv2/bn")
        if stride != 1:
            self.downsample = tf.keras.Sequential(name="shortcut")
            self.downsample.add(
                tf.keras.layers.Conv2D(
                    filters=filter_num,
                    kernel_size=(1, 1),
                    strides=stride,
                    use_bias=use_bias,
                    name="conv"
                ))
            self.downsample.add(tf.keras.layers.BatchNormalization(name="bn"))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1, use_bias=True, name=None):
        super(BottleNeck, self).__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="conv1"
        )
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv1/bn")
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=use_bias,
            name="conv2"
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="conv2/bn")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filter_num * 4,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="conv3"
        )
        self.bn3 = tf.keras.layers.BatchNormalization(name="conv3/bn")

        self.downsample = tf.keras.Sequential(name="shortcut")
        self.downsample.add(
            tf.keras.layers.Conv2D(
                filters=filter_num * 4,
                kernel_size=(1, 1),
                strides=stride,
                use_bias=use_bias,
                name="conv"
            ))
        self.downsample.add(tf.keras.layers.BatchNormalization(name="bn"))

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, x]))

        return output


def make_basic_block_layer(filter_num,
                           blocks,
                           stride=1,
                           use_bias=True,
                           name=None):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(
        BasicBlock(filter_num, stride=stride, use_bias=use_bias,
                   name="block0"))

    for i in range(1, blocks):
        res_block.add(
            BasicBlock(filter_num,
                       stride=1,
                       use_bias=use_bias,
                       name=f"block{i}"))

    return res_block


def make_bottleneck_layer(filter_num,
                          blocks,
                          stride=1,
                          use_bias=True,
                          name=None):
    res_block = tf.keras.Sequential(name=name)
    res_block.add(
        BottleNeck(filter_num, stride=stride, use_bias=use_bias,
                   name="block0"))

    for i in range(1, blocks):
        res_block.add(
            BottleNeck(filter_num,
                       stride=1,
                       use_bias=use_bias,
                       name=f"block{i}"))

    return res_block
