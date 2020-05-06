import multiprocessing

import cv2
import numpy as np
from tensorpack import (BatchData, MultiThreadMapData, PrefetchDataZMQ,
                        dataset, imgaug)


def fbresnet_augmentor(target_shape, training=True):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if training:
        raise ValueError("Not implemented")
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224))
        ]

        if target_shape != 224:
            augmentors.append(imgaug.ResizeShortestEdge(target_shape, cv2.INTER_CUBIC))

    return augmentors


def get_imagenet_dataflow(
        datadir, batch_size,
        augmentors, mode='train'):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert mode in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    training = mode == 'train'
    cpu = min(30, multiprocessing.cpu_count())
    meta_dir = './ilsvrc_metadata'
    if training:
        raise ValueError("Not implemented")
    else:
        ds = dataset.ILSVRC12Files(datadir, mode, meta_dir=meta_dir,
                                   shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            im = np.array(im, np.float32) / 255.0
            return im, cls

        ds = MultiThreadMapData(ds, cpu, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds
