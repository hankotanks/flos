import torch

from typing import Dict as dict
from typing import List as list
from typing import Tuple as tuple

class AddOffsets(object):
    """
    Best as a pre_transform.
    NOTE: Batches are not handled.
    NOTE: the transformed data's offset can contain nan's when there are
    noise points which do not belong to any instance. these should be
    filtered in the loss.

    :param noise_iid: this instance id is ignored when calculating offsets.
    """

    def __init__(self, ignore_id: int = 0, verbose=True):
        self.ignore_id = ignore_id
        self.verbose = verbose

    def __call__(self, data: dict[str, torch.Tensor]):
        if self.verbose:
            print("applying transform - add offsets")
        pos = data["pos"]
        iids = data["instance"]
        offset = torch.full_like(pos, 0.0)
        unique_iids = iids.unique()
        unique_iids = unique_iids[unique_iids != self.ignore_id]

        for iid in unique_iids:
            iid_mask = (iids == iid).flatten()
            instance_pos = pos[iid_mask]
            instance_center = instance_pos.mean(dim=-2)
            instance_offset = instance_center - instance_pos
            offset[iid_mask] = instance_offset

        data["offset"] = offset
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_iid={self.noise_iid})"
