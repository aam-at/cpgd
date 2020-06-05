import logging
import os
import pickle
import sys
from functools import partial

import jax
import numpy as onp

from .utils import is_device_array


def find_starting_point(images, labels, args, x, label, logits, predict_class):
    strategy = args.nth_likely_class_starting_point
    if strategy is None:
        return find_starting_point_simple_strategy(images, labels, x, label, predict_class)
    return find_starting_point_likely_class_strategy(images, labels, x, label, logits, predict_class, nth=strategy)


def find_starting_point_simple_strategy(images, labels, x, label, predict_class):
    """returns the image in images that is closest to x that has a
    different label and predicted class than the provided label of x"""

    assert x.shape[0] == 1

    assert not is_device_array(x)
    assert not is_device_array(label)

    assert not is_device_array(images)
    assert not is_device_array(labels)
    assert not is_device_array(x)
    assert not is_device_array(label)

    # filter those with the same label
    images = images[labels != label]

    # get closest images from other classes
    diff = images - x
    diff = diff.reshape((diff.shape[0], -1))
    diff = onp.square(diff).sum(axis=-1)
    diff = onp.argsort(diff)
    assert diff.ndim == 1

    for j, index in enumerate(diff):
        logging.info(f'trying {j + 1}. candidate ({index})')
        candidate = images[index][onp.newaxis]
        class_ = jax.device_get(predict_class(candidate).squeeze(axis=0))
        logging.info(f'label = {label}, candidate class = {class_}')
        if class_ != label:
            return candidate, class_


def find_starting_point_likely_class_strategy(images, labels, x, label, logits, predict_class, *, nth):
    assert x.shape[0] == 1

    assert not is_device_array(x)
    assert not is_device_array(label)

    assert not is_device_array(images)
    assert not is_device_array(labels)
    assert not is_device_array(x)
    assert not is_device_array(label)

    # determine nth likely class
    logits = logits.squeeze(axis=0)
    ordered_classes = onp.argsort(logits)
    assert ordered_classes[-1] == label

    assert 2 <= nth <= len(logits)
    nth_class = ordered_classes[-nth]

    # select those from the nth most likely  class
    images = images[labels == nth_class]

    # get closest images from other classes
    diff = images - x
    diff = diff.reshape((diff.shape[0], -1))
    diff = onp.square(diff).sum(axis=-1)
    diff = onp.argsort(diff)
    assert diff.ndim == 1

    for j, index in enumerate(diff):
        logging.info(f'trying {j + 1}. candidate ({index})')
        candidate = images[index][onp.newaxis]
        class_ = jax.device_get(predict_class(candidate).squeeze(axis=0))
        logging.info(f'label = {label}, candidate class = {class_}')
        if class_ != label:
            return candidate, class_
