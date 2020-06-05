"""A modified version of stax that supports tracking of intermediate
activations, in particular the inputs to non-affine layers."""
import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   Relu, MaxPool)


def affine(layer_fun):
    """Decorator that turns a layer into one that's compatible with tracking
    of additional outputs."""

    # @functools.wraps(layer_fun)
    def wrapper(*args, **kwargs):
        init_fun, apply_fun = layer_fun(*args, **kwargs)

        def new_apply_fun(*args, **kwargs):
            return apply_fun(*args, **kwargs), ()

        return init_fun, new_apply_fun

    return wrapper


def affine_no_params(layer):
    """Decorator that turns a layer into one that's compatible with tracking
    of additional outputs."""
    init_fun, apply_fun = layer

    def new_apply_fun(*args, **kwargs):
        return apply_fun(*args, **kwargs), ()

    return init_fun, new_apply_fun


def track_input_no_params(layer):
    init_fun, apply_fun = layer

    def new_apply_fun(params, inputs, rng=None):
        return apply_fun(params, inputs, rng=rng), (inputs, )

    return init_fun, new_apply_fun


def serial(*layers):
    """Like stax.serial but separately tracks additional outputs
    for each layer."""
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)
    rng = random.PRNGKey(0)

    def init_fun(input_shape):
        params = []
        for init_fun in init_funs:
            input_shape, param = init_fun(rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, rng=None):
        rngs = random.split(rng,
                            nlayers) if rng is not None else (None, ) * nlayers
        additional_outputs = []
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs, additional_output = fun(param, inputs, rng=rng)
            additional_outputs.append(additional_output)
        return inputs, additional_outputs

    return init_fun, apply_fun


AvgPool = affine(AvgPool)
MaxPool = affine(MaxPool)
BatchNorm = affine(BatchNorm)
Conv = affine(Conv)
Dense = affine(Dense)
FanInSum = affine_no_params(FanInSum)
FanOut = affine(FanOut)
Flatten = affine_no_params(Flatten)
GeneralConv = affine(GeneralConv)
Identity = affine_no_params(Identity)
Relu = track_input_no_params(Relu)
