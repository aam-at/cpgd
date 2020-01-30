import tensorflow as tf
from absl import flags

from nn import SNConv2D, SNDense
from utils import add_default_end_points, change_default_args

FLAGS = flags.FLAGS


def register_model_flags(model_name="model",
                         model="lenet5",
                         activation="relu",
                         feature_layer_dims="1200-1200-1200",
                         classifier_layer_dims="",
                         layer_type="standard",
                         prefix=''):
    # model parameters
    flags.DEFINE_string("%smodel_name" % prefix, model_name,
                        "name of the model")
    flags.DEFINE_string("%smodel" % prefix, model,
                        "model name (mlp or lenet5)")
    flags.DEFINE_string("%sactivation" % prefix, activation,
                        "activation function")
    flags.DEFINE_string("%sfeature_layer_dims" % prefix, feature_layer_dims,
                        "dimensions of feature network (shared between classifiers)")
    flags.DEFINE_string("%sclassifier_layer_dims" % prefix, classifier_layer_dims,
                        "dimensions of classifier network")
    flags.DEFINE_string("%slayer_type" % prefix, layer_type, "layer type")


class MLP(tf.keras.Model):
    def __init__(self,
                 feature_layer_dims,
                 classifier_layer_dims,
                 num_classes,
                 use_bias=True,
                 layer_type="standard",
                 activation='relu'):
        super(MLP, self).__init__()
        self.feature_layer_dims = feature_layer_dims
        self.classifier_layer_dims = classifier_layer_dims
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.layer_type = layer_type
        self.activation = activation

    def _mlp_transform(self, z, layer_dims, scope=None):
        if self.layer_type == 'standard':
            dense = tf.keras.layers.Dense
        elif self.layer_type == 'spectral':
            dense = SNDense
        else:
            raise ValueError("Invalid layer type '{}'".format(self.layer_type))
        dense = change_default_args(dense, use_bias=self.use_bias)
        act = tf.keras.activations.get(self.activation)
        if len(layer_dims) > 0:
            for i, layer_size in enumerate(layer_dims):
                dense_name = scope + '/dense_%d' % i
                act_name = scope + '/act_%d' % i
                assert layer_size >= 0
                h = dense(layer_size, name=dense_name)(z)
                z = tf.keras.layers.Lambda(
                    lambda x: act(x), name=act_name)(h)
        return z

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        # define functional computation graph
        xflt = tf.keras.layers.Flatten()(x)
        # TODO: name_scope does not work in tf2.0
        h = self._mlp_transform(xflt, self.feature_layer_dims, scope="features")
        z = self._mlp_transform(h, self.classifier_layer_dims, scope="classifier")
        logits = tf.keras.layers.Dense(self.num_classes, activation=None,
                                       name='logits')(z)
        outputs = {'features': h, 'logits': logits}
        self.model = tf.keras.Model(inputs=x, outputs=outputs)

    @property
    def feature_trainable_variables(self):
        variables = []
        for var in self.trainable_variables:
            if 'features' in var.name:
                variables.append(var)
        return variables

    def classifier_trainable_variables(self):
        variables = []
        for var in self.trainable_variables:
            if 'classifier' in var.name:
                variables.append(var)
        return variables

    def call(self, inputs, training=True):
        outputs = self.model(inputs, training=training)
        return add_default_end_points(outputs)


class TradesCNN(tf.keras.Model):
    def __init__(self, drop=0.5):
        super(TradesCNN, self).__init__()
        self.drop = drop

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        conv = tf.keras.layers.Conv2D
        dense = tf.keras.layers.Dense
        # configure layers
        conv = change_default_args(conv,
                                   data_format='channels_first')
        max_pool = change_default_args(tf.keras.layers.MaxPool2D,
                                       data_format='channels_first')
        dense = change_default_args(dense)
        act = change_default_args(tf.keras.layers.Activation, activation='relu')
        # define functional computation graph
        with tf.init_scope():
            z = conv(32, 3)(x)
            z = act()(z)
            z = conv(32, 3)(z)
            z = act()(z)
            z = max_pool(2)(z)
            z = conv(64, 3)(z)
            z = act()(z)
            z = conv(64, 3)(z)
            z = act()(z)
            z = max_pool(2)(z)
            z = tf.keras.layers.Flatten()(z)
            # classifier
            h = dense(200)(z)
            h = act()(h)
            # h = tf.keras.layers.Dropout(self.drop)(h)
            h = dense(200)(h)
            h = act()(h)
            logits = tf.keras.layers.Dense(10, activation=None)(h)
        self.model = tf.keras.Model(inputs=x, outputs=[z, logits])

    def call(self, inputs, training=True):
        z, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': z, 'logits': logits})


class MadryCNN(tf.keras.Model):
    def __init__(self):
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
                                   padding='same',
                                   data_format='channels_last')
        max_pool = change_default_args(tf.keras.layers.MaxPool2D,
                                       data_format='channels_last')
        dense = change_default_args(dense)
        activation = tf.keras.activations.get('relu')
        # define functional computation graph
        with tf.init_scope():
            z = conv(32, 5)(x)
            z = max_pool(2)(z)
            z = conv(64, 5)(z)
            z = max_pool(2)(z)
            z = tf.keras.layers.Flatten()(z)
            h = dense(1024)(z)
            z = tf.keras.layers.Lambda(lambda x: activation(x))(h)
            logits = tf.keras.layers.Dense(10, activation=None)(z)
        self.model = tf.keras.Model(inputs=x, outputs=[h, logits])

    def call(self, inputs, training=True):
        h, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': h, 'logits': logits})


class Lenet5(tf.keras.Model):
    def __init__(self, num_classes, activation, layer_type="standard"):
        super(Lenet5, self).__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.layer_type = layer_type

    def build(self, inputs_shape):
        # configure inputs
        x_shape = inputs_shape
        x = tf.keras.layers.Input(shape=x_shape[1:], name='x')
        if self.layer_type == 'standard':
            conv = tf.keras.layers.Conv2D
            dense = tf.keras.layers.Dense
        elif self.layer_type == 'spectral':
            conv = SNConv2D
            dense = SNDense
        else:
            raise ValueError("Invalid layer type '{}'".format(self.layer_type))
        # configure layers
        conv = change_default_args(conv,
                                   activation=self.activation,
                                   data_format='channels_first')
        max_pool = change_default_args(tf.keras.layers.MaxPool2D,
                                       data_format='channels_first')
        dense = change_default_args(dense)
        activation = tf.keras.activations.get(self.activation)
        # define functional computation graph
        with tf.init_scope():
            z = conv(32, 5)(x)
            z = max_pool(2)(z)
            z = conv(64, 5)(z)
            z = max_pool(2)(z)
            z = tf.keras.layers.Flatten()(z)
            h = dense(512)(z)
            z = tf.keras.layers.Lambda(lambda x: activation(x))(h)
            logits = tf.keras.layers.Dense(self.num_classes,
                                           activation=None)(z)
        self.model = tf.keras.Model(inputs=x, outputs=[h, logits])

    def call(self, inputs, training=True):
        h, logits = self.model(inputs, training=training)
        return add_default_end_points({'features': h, 'logits': logits})


def create_model(num_classes, model=None, prefix=''):
    if model is None:
        model = getattr(FLAGS, '%smodel' % prefix)
    activation = getattr(FLAGS, '%sactivation' % prefix)
    layer_type = getattr(FLAGS, "%slayer_type" % prefix)
    if model == 'mlp':
        feature_layer_dims = getattr(FLAGS, '%sfeature_layer_dims' % prefix)
        feature_layer_dims = [int(dim) for dim in feature_layer_dims.split("-")]
        classifier_layer_dims = getattr(FLAGS,
                                        '%sclassifier_layer_dims' % prefix)
        classifier_layer_dims = [
            int(dim) for dim in classifier_layer_dims.split("-") if dim != ""
        ]
        return MLP(feature_layer_dims,
                   classifier_layer_dims,
                   num_classes,
                   activation=activation,
                   layer_type=layer_type)
    elif model == 'lenet5':
        return Lenet5(num_classes, activation=activation, layer_type=layer_type)
    elif model == 'trades_cnn':
        return TradesCNN(num_classes, layer_type=layer_type)
    else:
        raise ValueError
