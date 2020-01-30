import numpy as np
import tensorflow as tf


def load_mnist(validation_size=10000, train_indices=False, shuffle=True,
               seed=123):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.cast[np.float32](np.expand_dims(x_train, 1))
    x_test = np.cast[np.float32](np.expand_dims(x_test, 1))
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


def make_input_pipeline(ds, shuffle=True, batch_size=128):
    if shuffle:
        ds = ds.shuffle(10 * batch_size)
    return ds.batch(batch_size, drop_remainder=True).prefetch(1)


def corrupt_labels(labels, num_classes, corrupt_prob=1.0, seed=12345):
    labels = labels.copy()
    rng = np.random.RandomState(seed)
    mask = rng.rand(len(labels)) <= corrupt_prob
    rnd_labels = rng.choice(num_classes, mask.sum())
    labels[mask] = rnd_labels
    return labels


# data utils
def batch_iterator(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    # reshape for NCHW
    inputs = inputs.transpose((0, 3, 1, 2))
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


def select_balanced_subset(X, y, num_classes=10, samples_per_class=10, seed=1):
    total_samples = num_classes * samples_per_class
    X_subset = np.zeros([total_samples] + list(X.shape[1:]), dtype=X.dtype)
    y_subset = np.zeros((total_samples, ), dtype=y.dtype)
    rng = np.random.RandomState(seed)
    for i in range(num_classes):
        yi_indices = np.where(y == i)[0]
        rng.shuffle(yi_indices)
        X_subset[samples_per_class * i:(i + 1) *
                 samples_per_class, ...] = X[yi_indices[:samples_per_class]]
        y_subset[samples_per_class * i:(i + 1) * samples_per_class] = i
    return X_subset, y_subset


def find_cluster_centers(X, y):
    classes = np.unique(y)
    class_centers = []
    for y_i in classes:
        X_i = X[np.where(y == y_i)]
        mu = np.mean(X_i, axis=0, keepdims=True)
        diff = np.sum((X_i - mu)**2, axis=(1, 2, 3))
        class_center = X_i[diff.argmin(axis=0)]
        class_centers.append(class_center)
    class_centers = np.asarray(class_centers)
    return class_centers, classes
