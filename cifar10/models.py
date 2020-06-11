import tensorflow as tf
import torch
from absl import flags
from torch import nn
from torch.nn import functional as F

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


class MadryCNNPt(torch.nn.Module):
    # Model trained using adversarial training with projected gradient attack
    def __init__(self, model_type='plain'):
        super(MadryCNNPt, self).__init__()
        use_bias = model_type == 'plain'
        self.c0 = nn.Conv2d(3, 96, 3, padding=1, bias=use_bias)
        self.c1 = nn.Conv2d(96, 96, 3, padding=1, bias=use_bias)
        self.c2 = nn.Conv2d(96, 192, 3, padding=1, stride=2, bias=use_bias)
        self.c3 = nn.Conv2d(192, 192, 3, padding=1, bias=use_bias)
        self.c4 = nn.Conv2d(192, 192, 3, padding=1, bias=use_bias)
        self.c5 = nn.Conv2d(192, 192, 3, padding=1, stride=2, bias=use_bias)
        self.c6 = nn.Conv2d(192, 192, 3, padding=1, bias=use_bias)
        self.c7 = nn.Conv2d(192, 384, 2, padding=1, stride=2, bias=use_bias)
        self.fc1 = nn.Linear(6144, 1200)
        self.fc2 = nn.Linear(1200, 10)

    def forward(self, x):
        o = F.relu(self.c0(x))
        o = F.relu(self.c1(x))
        o = F.relu(self.c2(x))
        o = F.relu(self.c3(x))
        o = F.relu(self.c4(x))
        o = F.relu(self.c5(x))
        o = F.relu(self.c6(x))
        o = F.relu(self.c7(x))
        # permute to be compatible with tensorflow
        o = o.permute(0, 2, 3, 1)
        o = o.reshape(x.shape[0], -1)
        o = F.relu(self.fc1(o))
        return self.fc2(o)
