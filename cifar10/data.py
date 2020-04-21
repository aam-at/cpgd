import numpy as np
import tensorflow.keras.datasets.cifar10 as cifar10


def load_cifar10(validation_size=10000,
                 shuffle=True,
                 data_format="NHWC",
                 seed=123):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.cast[np.float32](x_train)
    x_test = np.cast[np.float32](x_test)
    y_train = np.reshape(np.cast[np.int64](y_train), (-1, ))
    y_test = np.reshape(np.cast[np.int64](y_test), (-1, ))

    def x_transform(x):
        if data_format == "NHWC":
            return x
        elif data_format == "NCHW":
            return np.transpose(x, (0, 3, 1, 2))
        else:
            raise ValueError

    # split train/validation
    total_train_examples = x_train.shape[0]
    indices = np.arange(total_train_examples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    train_examples = total_train_examples - validation_size
    x_val, y_val = x_train[train_examples:], y_train[train_examples:]
    x_train, y_train = x_train[:train_examples], y_train[:train_examples]
    return ((x_transform(x_train), y_train), (x_transform(x_val), y_val),
            (x_transform(x_test), y_test))
