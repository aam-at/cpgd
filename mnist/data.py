import numpy as np
import tensorflow.keras.datasets.mnist as mnist


def load_mnist(validation_size=10000,
               shuffle=True,
               data_format="NHWC",
               seed=123):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.cast[np.float32](np.expand_dims(x_train, 1))
    x_test = np.cast[np.float32](np.expand_dims(x_test, 1))
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = np.cast[np.int64](y_train)
    y_test = np.cast[np.int64](y_test)

    def x_transform(x):
        if data_format == "NHWC":
            return np.transpose(x, (0, 2, 3, 1))
        elif data_format == "NCHW":
            return x
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


def batch_iterator(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    n_samples = inputs.shape[0]
    if shuffle:
        # Shuffles indicies of training data, so we can draw batches
        # from random indicies instead of shuffling whole data
        indx = np.random.permutation(range(n_samples))
    else:
        indx = range(n_samples)
    for i in range((n_samples + batchsize - 1) // batchsize):
        sl = slice(i * batchsize, (i + 1) * batchsize)
        X_batch = inputs[indx[sl]]
        y_batch = targets[indx[sl]]
        yield X_batch, y_batch
