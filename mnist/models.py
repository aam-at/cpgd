import tensorflow as tf
import torch
from torch import nn
from torch.nn import functional as F


class MadryCNNTf(tf.keras.Model):
    # Model trained using adversarial training with projected gradient attack
    def __init__(self, name="", **kwargs):
        super(MadryCNNTf, self).__init__(name=name, **kwargs)

    def build(self, inputs_shape):
        from lib.tf_utils import change_default_args
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
                                       padding='valid',
                                       data_format='channels_last')
        dense = change_default_args(dense, activation=None)
        act = change_default_args(tf.keras.layers.Activation, activation='relu')
        # define functional computation graph
        with tf.init_scope():
            z = conv(32, 5)(x)
            z = max_pool(2)(z)
            z = conv(64, 5)(z)
            z = max_pool(2)(z)
            # classifier
            z = tf.keras.layers.Flatten()(z)
            h = dense(1024)(z)
            z = act()(h)
            logits = dense(10)(z)
        self.model = tf.keras.Model(inputs=x, outputs=[h, logits])
        super(MadryCNNTf, self).build([inputs_shape])

    def call(self, inputs, training=True):
        from lib.tf_utils import add_default_end_points
        h, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': h, 'logits': logits})


class MadryCNNPt(torch.nn.Module):
    # Model trained using adversarial training with projected gradient attack
    def __init__(self, wrap_outputs=True):
        super(MadryCNNPt, self).__init__()
        self.c1 = nn.Conv2d(1, 32, 5, padding=2)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(32, 64, 5, padding=2)
        self.m2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.wrap_outputs = wrap_outputs

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, wrap_outputs=None):
        from lib.pt_utils import add_default_end_points
        if wrap_outputs is None:
            wrap_outputs = self.wrap_outputs
        o = F.relu(self.c1(x))
        o = self.m1(o)
        o = F.relu(self.c2(o))
        o = self.m2(o)
        # permute to be compatible with tensorflow
        o = o.permute(0, 2, 3, 1)
        o = o.reshape(x.shape[0], -1)
        o = F.relu(self.fc1(o))
        logits = self.fc2(o)
        if wrap_outputs:
            return add_default_end_points({'logits': logits})
        else:
            return logits
