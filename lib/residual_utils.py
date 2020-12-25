import functools

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.keras import backend

BatchNorm2dTf = functools.partial(tf.keras.layers.BatchNormalization,
                                  epsilon=1e-5,
                                  momentum=0.9)
BatchNorm2dPt = functools.partial(nn.BatchNorm2d, eps=1e-5, momentum=0.1)


class BottleNeckTf(tf.keras.layers.Layer):
    expansion = 4

    def __init__(self,
                 filter_num,
                 stride=1,
                 conv_shortcut=True,
                 use_bias=True,
                 activation=tf.nn.relu,
                 name=None):
        super(BottleNeckTf, self).__init__(name=name)
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
        self.bn1 = BatchNorm2dTf(name="conv1/bn")
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding="same",
            use_bias=use_bias,
            name="conv2",
        )
        self.bn2 = BatchNorm2dTf(name="conv2/bn")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filter_num * self.expansion,
            kernel_size=(1, 1),
            strides=1,
            padding="same",
            use_bias=use_bias,
            name="conv3",
        )
        self.bn3 = BatchNorm2dTf(name="conv3/bn")

        if conv_shortcut:
            self.downsample = tf.keras.Sequential(name="shortcut")
            self.downsample.add(
                tf.keras.layers.Conv2D(
                    filters=filter_num * self.expansion,
                    kernel_size=(1, 1),
                    strides=stride,
                    use_bias=use_bias,
                    name="conv",
                ))
            self.downsample.add(BatchNorm2dTf(name="bn"))
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


class ResnetCNNTf(tf.keras.Model):
    def __init__(self):
        super(ResnetCNNTf, self).__init__()

    def _make_block(self, planes, blocks, stride=1, name=None):
        block = tf.keras.Sequential(name=name)
        block.add(
            BottleNeckTf(
                planes,
                stride=stride,
                conv_shortcut=True,
                use_bias=False,
                activation=tf.nn.relu,
                name="block0",
            ))

        for i in range(1, blocks):
            block.add(
                BottleNeckTf(
                    planes,
                    stride=1,
                    conv_shortcut=False,
                    use_bias=False,
                    activation=tf.nn.relu,
                    name=f"block{i}",
                ))

        return block

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name="x")
        bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

        # define functional computation graph
        with tf.init_scope():
            z = tf.keras.layers.Conv2D(64,
                                       7,
                                       strides=2,
                                       use_bias=False,
                                       padding="SAME",
                                       activation=None,
                                       name="conv0")(x)
            z = BatchNorm2dTf(axis=bn_axis, name="conv0/bn")(z)
            z = tf.keras.layers.ReLU()(z)
            z = tf.keras.layers.MaxPool2D(pool_size=3,
                                          strides=2,
                                          padding="SAME")(z)
            block1 = self._make_block(64, 3, stride=1, name="group0")
            z = block1(z)
            block2 = self._make_block(128, 4, stride=2, name="group1")
            z = block2(z)
            block3 = self._make_block(256, 6, stride=2, name="group2")
            z = block3(z)
            block4 = self._make_block(512, 3, stride=2, name="group3")
            z = block4(z)
            z = tf.keras.layers.GlobalAveragePooling2D()(z)
            logits = tf.keras.layers.Dense(1000)(z)
        self.model = tf.keras.Model(inputs=x, outputs=logits)

    def call(self, inputs, training=True):
        logits = self.model(inputs, training=training)
        return logits


class BottleneckPt(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 base_width=64):
        super(BottleneckPt, self).__init__()
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=None)
        self.bn1 = BatchNorm2dPt(width)
        self.conv2 = nn.Conv2d(width,
                               width,
                               3,
                               stride=stride,
                               padding=1 if stride == 1 else 0,
                               bias=None)
        self.bn2 = BatchNorm2dPt(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=None)
        self.bn3 = BatchNorm2dPt(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.conv2.stride[0] == 2:
            out = F.pad(out, (0, 1, 0, 1))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetCNNPt(nn.Module):
    def __init__(self):
        super(ResNetCNNPt, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3,
                               self.inplanes,
                               kernel_size=7,
                               stride=2,
                               padding=0,
                               bias=False)
        self.bn1 = BatchNorm2dPt(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1000)

    def _make_layer(self, planes, blocks, stride=1):
        conv_shortcut = nn.Sequential(
            nn.Conv2d(self.inplanes,
                      planes * 4,
                      kernel_size=1,
                      stride=stride,
                      padding=0,
                      bias=None), BatchNorm2dPt(planes * 4))

        layers = []
        layers.append(
            BottleneckPt(self.inplanes, planes, stride, conv_shortcut, 64))
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(
                BottleneckPt(self.inplanes,
                             planes,
                             base_width=64))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # padding for tensorflow "SAME" behavior
        x = F.pad(x, (2, 3, 2, 3))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # padding for tensorflow "SAME" behavior
        x = F.pad(x, (0, 1, 0, 1))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
