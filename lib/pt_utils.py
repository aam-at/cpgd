from __future__ import absolute_import, division, print_function

import ignite
import numpy as np
import torch


class PatchedAverage(ignite.metrics.Average):
    def __call__(self, value):
        value = value.view(-1, 1)
        self.update(value)

    def reset_states(self):
        self.reset()

    def result(self):
        return self.compute()


class MetricsDictionary(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            dict.__setitem__(self, key, PatchedAverage())
            return self.__getitem__(key)


def to_torch(*args, cuda=True):
    torch_tensors = [torch.from_numpy(np.asarray(a)) for a in args]
    return [t.cuda() if cuda else t for t in torch_tensors]


def prediction(prob, dim=-1):
    return torch.argmax(prob, dim=dim)


def add_default_end_points(end_points):
    logits = end_points["logits"]
    predictions = prediction(logits)
    prob = torch.softmax(logits, dim=-1)
    log_prob = torch.log_softmax(logits, dim=-1)
    conf = torch.argmax(prob, dim=-1)
    end_points.update({
        "pred": predictions,
        "prob": prob,
        "log_prob": log_prob,
        "conf": conf
    })
    return end_points


def l0_metric(x, dim=-1, keepdim=False):
    return (x.abs() > 0).type(torch.float32).sum(dim=dim, keepdim=keepdim)


def l0_pixel_metric(u, channel_dim=-1, keepdim=False):
    u_c = torch.argmax(u, dim=channel_dim)
    return l0_metric(u_c, keepdim=keepdim)


def li_metric(x, dim=-1, keepdim=False):
    return x.abs().max(dim=dim, keepdim=keepdim)[0]


def l1_metric(x, dim=-1, keepdim=False):
    """L1 metric
    """
    return x.abs().sum(dim=dim, keepdim=keepdim)


def l2_metric(x, dim=-1, keepdim=False):
    """L2 metric
    """
    square_sum = x.square().sum(dim=dim, keepdim=keepdim)
    return torch.sqrt(square_sum)
