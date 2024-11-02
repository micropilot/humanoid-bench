from jax import numpy as jp
from jax import lax
import warnings

_DEFAULT_VALUE_AT_MARGIN = 0.1

def _sigmoids(x, value_at_1, sigmoid):
    if sigmoid == 'gaussian':
        scale = jp.sqrt(-2 * jp.log(value_at_1))
        return jp.exp(-0.5 * (x * scale) ** 2)
    elif sigmoid == 'hyperbolic':
        scale = jp.arccosh(1 / value_at_1)
        return 1 / jp.cosh(x * scale)
    elif sigmoid == 'long_tail':
        scale = jp.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)
    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (jp.abs(x) * scale + 1)
    elif sigmoid == 'cosine':
        scale = jp.arccos(2 * value_at_1 - 1) / jp.pi
        scaled_x = x * scale
        return jp.where(jp.abs(scaled_x) < 1, (1 + jp.cos(jp.pi * scaled_x)) / 2, 0.0)
    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return jp.where(jp.abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    elif sigmoid == 'quadratic':
        scale = jp.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return jp.where(jp.abs(scaled_x) < 1, 1 - scaled_x ** 2, 0.0)
    elif sigmoid == 'tanh_squared':
        scale = jp.arctanh(jp.sqrt(1 - value_at_1))
        return 1 - jp.tanh(x * scale) ** 2
    else:
        raise ValueError(f'Unknown sigmoid type {sigmoid}')

def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian', value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    lower, upper = bounds

    # Remove the dynamic check for tracing compatibility
    assert lower <= upper, "Lower bound must be <= upper bound"

    # Calculate the tolerance value
    in_bounds = jp.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = jp.where(in_bounds, 1.0, 0.0)
    else:
        d = jp.where(x < lower, lower - x, x - upper) / margin
        value = jp.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value
