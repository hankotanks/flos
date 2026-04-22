from torch import Tensor

from typing import Dict as dict
from typing import List as list
from typing import Tuple as tuple

class CenterGlobal(object):
    """
    Center the positions in euclidean space.
    Assumes the data['pos'] is a ...xNx3 shape tensor.
    Best as a pre_transform.
    """

    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, data: dict[str, Tensor]):
        if self.verbose:
            print("applying transform - centering positions")
        data["pos"] -= data["pos"].mean(axis=-2).unsqueeze(-2)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
