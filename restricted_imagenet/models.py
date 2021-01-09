import tensorflow as tf
import torch
from lib.residual_utils import ResNetCNNPt, ResnetCNNTf


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
        self.backbone = ResnetCNNTf()
        self.cast = tf.keras.layers.Activation('linear', dtype=tf.float32)

    @staticmethod
    def image_preprocess(image, bgr=True):
        mean = [0.485, 0.456, 0.406]  # rgb
        std = [0.229, 0.224, 0.225]
        if bgr:
            mean = mean[::-1]
            std = std[::-1]
        image_mean = tf.constant(mean, dtype=image.dtype)
        image_std = tf.constant(std, dtype=image.dtype)
        image = (image - image_mean) / image_std
        return image

    def call(self, inputs, training=True):
        from lib.tf_utils import add_default_end_points
        inputs = self.image_preprocess(inputs)
        logits = self.backbone(inputs, training=training)
        logits = self.cast(logits)
        num_labels = len(TsiprasCNN.LABEL_RANGES)
        return add_default_end_points({"logits": logits[:, :num_labels]})


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
        self.backbone = ResNetCNNPt()
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
