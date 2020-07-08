import tensorflow as tf
import torch
from tensorflow.python.keras import backend
from torchvision.models.resnet import resnet50

from residual_utils import make_bottleneck_layer


class TsiprasCNN(tf.keras.Model):
    LABEL_RANGES = [
        (151, 268),
        (281, 285),
        (30, 32),
        (33, 37),
        (80, 100),
        (365, 382),
        (389, 397),
        (118, 121),
        (300, 319),
    ]

    # Imagenet robust model Tsipras et al
    def __init__(self):
        super(TsiprasCNN, self).__init__()
        self.backbone = ResnetCNN()

    @staticmethod
    def image_preprocess(image, bgr=True):
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_std = tf.constant(std, dtype=tf.float32)
        image = (image - image_mean) / image_std
        return image

    def call(self, inputs, training=True):
        from lib.tf_utils import add_default_end_points
        inputs = self.image_preprocess(inputs)
        logits = self.backbone(inputs, training=training)
        num_labels = len(TsiprasCNN.LABEL_RANGES)
        return add_default_end_points({"logits": logits[:, :num_labels]})


class ResnetCNN(tf.keras.Model):
    def __init__(self):
        super(ResnetCNN, self).__init__()

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
                                       name="conv0")(x)
            z = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                                   epsilon=1e-5,
                                                   name="conv0/bn")(z)
            z = tf.keras.layers.ReLU()(z)
            z = tf.keras.layers.MaxPool2D(pool_size=3,
                                          strides=2,
                                          padding="SAME")(z)
            block1 = make_bottleneck_layer(64,
                                           3,
                                           use_bias=False,
                                           stride=1,
                                           name="group0")
            z = block1(z)
            block2 = make_bottleneck_layer(128,
                                           4,
                                           use_bias=False,
                                           stride=2,
                                           name="group1")
            z = block2(z)
            block3 = make_bottleneck_layer(256,
                                           6,
                                           use_bias=False,
                                           stride=2,
                                           name="group2")
            z = block3(z)
            block4 = make_bottleneck_layer(512,
                                           3,
                                           use_bias=False,
                                           stride=2,
                                           name="group3")
            z = block4(z)
            z = tf.keras.layers.GlobalAveragePooling2D()(z)
            logits = tf.keras.layers.Dense(1000)(z)
        self.model = tf.keras.Model(inputs=x, outputs=logits)

    def call(self, inputs, training=True):
        logits = self.model(inputs, training=training)
        return logits


class TsiprasCNNPt(torch.nn.Module):
    LABEL_RANGES = [
        (151, 268),
        (281, 285),
        (30, 32),
        (33, 37),
        (80, 100),
        (365, 382),
        (389, 397),
        (118, 121),
        (300, 319),
    ]

    # Imagenet robust model Tsipras et al
    def __init__(self, data_format="NHWC", wrap_outputs=True):
        super(TsiprasCNNPt, self).__init__()
        self.backbone = resnet50(pretrained=False)
        self.data_format = data_format
        self.wrap_outputs = wrap_outputs

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def image_preprocess(image, bgr=True):
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = torch.tensor(mean, dtype=torch.float32).to(image.device)
        image_std = torch.tensor(std, dtype=torch.float32).to(image.device)
        image = (image - image_mean) / image_std
        return image

    def forward(self, x, wrap_outputs=None):
        from lib.pt_utils import add_default_end_points
        if wrap_outputs is None:
            wrap_outputs = self.wrap_outputs
        x = self.image_preprocess(x)
        if self.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)
        logits = self.backbone(x)
        num_labels = len(TsiprasCNNPt.LABEL_RANGES)
        logits = logits[:, :num_labels]
        if wrap_outputs:
            return add_default_end_points({'logits': logits})
        else:
            return logits
