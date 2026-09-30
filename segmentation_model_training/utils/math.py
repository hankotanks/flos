import typing

import numpy.typing

try:
    import torch
    TORCH_EXISTS = True
except ImportError:
    TORCH_EXISTS = False

def lmap(v: numpy.typing.NDArray, 
    x: typing.Union[typing.Tuple[typing.Any, typing.Any], numpy.typing.NDArray], 
    y: typing.Union[typing.Tuple[typing.Any, typing.Any], numpy.typing.NDArray]):
    """Linear map of value(s) v with range x to desired range y."""
    # not adding an epsilon to prevent zero div, can add later
    assert x[1] != x[0], "Division by zero, check x range"
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

def safe_divide(num, denom, eps: float = 1e-8):
    """prevents division by zero by clipping denom to eps"""
    if denom < eps:
        denom = eps
    return num / denom


def vector_safe_divide(num, denom, eps: float = 1e-8):
    """prevents division by zero by clipping denom to eps. vector version"""
    denom[denom < eps] = eps
    return num / denom


if TORCH_EXISTS:
    def torch_safe_divide(num: torch.Tensor, denom: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Safe division, by preventing division by zero.
        modified torchmetrics version, sets denom to eps instead of 1 when its low.
        Additionally casts to float if input is not already to secure backwards compatibility.
        """
        denom[denom < eps] = eps
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        return num / denom

else:
    def torch_safe_divide(*args, **kwargs):
        raise Exception("torch is not installed")