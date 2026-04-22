from itertools import repeat

import torch
from torch import Tensor

from typing import Dict as dict
from typing import List as list
from typing import Tuple as tuple
from typing import Union

Number = Union[int, float]
Number_Sequence = Union[list[Number], tuple[Number]]

class RandomTranslateGlobal(object):
    """
    Translates shapes by randomly sampled translation values
    within a given interval. This translation happens for the entire shape,
    retaining local relationships.
    Best as a batch transform on the gpu.

    :param translate: Maximum translation in each dimension, defining the
    range (-translate, +translate) per axis to sample from. If a number
    instead of a sequence, the same range is used for each dimension.
    """

    def __init__(self, translate: Union[Number, Number_Sequence]):
        self.translate = translate

    def __call__(self, data: dict[str, Tensor]):
        euc_dim = data["pos"].shape[-1]
        translate = self.translate
        if isinstance(translate, (int, float)):
            translate = list(repeat(translate, times=euc_dim))
        assert len(translate) == euc_dim

        transform = []
        for dim in range(euc_dim):
            transform.append(
                data["pos"]
                .new_empty(1)
                .uniform_(-abs(translate[dim]), abs(translate[dim]))
            )
        data["pos"] = data["pos"] + torch.stack(transform, dim=-1)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.translate})"
