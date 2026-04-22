import numpy as np
import torch
import torch.linalg as LA
from torch import Tensor
import random

from typing import Dict as dict
from typing import List as list
from typing import Tuple as tuple
from typing import Union

Number = Union[int, float]
Number_Sequence = Union[list[Number], tuple[Number, Number]]

class RandomUniformScale(object):
    """
    Scales positions by a randomly sampled factor `s` within a
    given interval.
    Also scales any vector information like normals and offsets if they
    exist.
    Best as a batch transform on the gpu.

    :param scales: interval [a, b] from which scale is randomly sampled.
    :param scale_vectors: scales any vector information like `normal` and
    `offset` if they exist (with those keys).
    """

    def __init__(
        self,
        scales: Number_Sequence,
        scale_vectors: bool = True,
        minkowski=False,
        verbose=False,
    ):
        "minkowski batch is treated differently for pos or coords rotation"
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.scale_vectors = scale_vectors
        self.minkowski = minkowski
        self.verbose = verbose

    def __call__(self, data: dict[str, Tensor]):
        scale = random.uniform(*self.scales)
        if self.minkowski:
            data["pos"][:, 1:] = data["pos"][:, 1:] * scale
        else:
            data["pos"] = data["pos"] * scale

        if self.scale_vectors and "normal" in data:
            raise NotImplementedError(
                "normals have some special consideration maybe. see deltaconv for weirdness"
            )
        if self.scale_vectors and "offset" in data:
            data["offset"] = data["offset"] * scale

        if self.verbose:
            print("Random Scale transform applied")
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.scales})"
