import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets.cifar10 as cifar10


def load_cifar10(validation_size=10000, train_indices=False, shuffle=True,
                 seed=123):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = np.cast[np.int64](y_train)
    y_test = np.cast[np.int64](y_test)
    # split train/validation
    total_train_examples = x_train.shape[0]
    indices = np.arange(total_train_examples)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    train_examples = total_train_examples - validation_size
    x_val, y_val = x_train[train_examples:], y_train[train_examples:]
    x_train, y_train = x_train[:train_examples], y_train[:train_examples]
    if train_indices:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train, np.arange(x_train.shape[0], dtype=np.int64)))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    return train_ds, val_ds, test_ds
