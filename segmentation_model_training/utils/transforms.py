import typing
import random
import itertools
import math

import torch

Number = typing.Union[int, float]
NumberSequence = typing.Union[typing.List[Number], typing.Tuple[Number, Number]]

class Transform(object):
    def __init__(self, verbose = False):
        self.verbose = verbose

    def __call__(self, data: typing.Dict[str, torch.Tensor]):
        if self.verbose: print(f"applying transform: {self.__class__.__name__}")
        return self._apply(data)
    
    def _apply(self, data: typing.Dict[str, torch.Tensor]):
        raise NotImplementedError(f"{self.__class__.__name__} must implement _apply")

    def __repr__(self):
        return f"transform::{self.__class__.__name__}"
    
class AddOffsets(Transform):
    """
    Best as a pre_transform.
    NOTE: Batches are not handled.
    NOTE: the transformed data's offset can contain nan's when there are
    noise points which do not belong to any instance. these should be
    filtered in the loss.

    :param noise_iid: this instance id is ignored when calculating offsets.
    """
    def __init__(self, ignore_id: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.ignore_id = ignore_id

    def _apply(self, data: typing.Dict[str, torch.Tensor]):
        pos = data["pos"]
        iids = data["instance"]
        offset = torch.full_like(pos, 0.0)
        unique_iids = iids.unique()
        unique_iids = unique_iids[unique_iids != self.ignore_id]
        for iid in unique_iids:
            iid_mask = (iids == iid).flatten()
            instance_pos = pos[iid_mask]
            instance_center = instance_pos.mean(dim = -2)
            instance_offset = instance_center - instance_pos
            offset[iid_mask] = instance_offset
        data["offset"] = offset
        return data

    def __repr__(self) -> str:
        return f"{super().__repr__()}(noise_iid={self.noise_iid})"

    
class CenterGlobal(Transform):
    """
    Center the positions in euclidean space.
    Assumes the data['pos'] is a ...xNx3 shape tensor.
    Best as a pre_transform.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _apply(self, data: typing.Dict[str, torch.Tensor]):
        data["pos"] -= data["pos"].mean(axis = -2).unsqueeze(-2)
        return data

class RandomRotate(Transform):
    """
    Rotates positions around a specific axis by a randomly sampled
    angle within a given interval.
    Best as a batch transform on the gpu.

    :params degrees: Rotation interval from which the rotation
    angle is sampled. If a number instead of a tuple, the interval is
    given by [-degrees, degrees].
    :param axis: The rotation axis. 0 - x, 1 - y, 2 - z.
    :param rotate_vectors: rotates any vector information like `normal` and
    `offset` if they exist (with those keys).
    """
    def __init__(self, degrees: typing.Union[typing.Tuple[Number], Number] = 180, axis: int = 2, rotate_vectors = True, **kwargs):
        super().__init__(**kwargs)
        "minkowski batch is treated differently for pos or coords rotation"
        if isinstance(degrees, (int, float)):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis
        self.rotate_vectors = rotate_vectors

    def _apply(self, data: typing.Dict[str, torch.Tensor]):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        if data["pos"].shape[-1] == 2:
            R_matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                R_matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                R_matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                R_matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]

        R_matrix = torch.Tensor(R_matrix).to(data["pos"].dtype).to(data["pos"].device)
        data["pos"][:, 1:] = torch.matmul(data["pos"][:, 1:], R_matrix.T)
        if self.rotate_vectors and "normal" in data:
            data["normal"] = torch.matmul(data["normal"], R_matrix.T)
        if self.rotate_vectors and "offset" in data:
            data["offset"] = torch.matmul(data["offset"], R_matrix.T)
        return data

    def __repr__(self):
        return f"{super().__repr__()}({self.degrees}, axis={self.axis})"

class RandomUniformScale(Transform):
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
    def __init__(self, scales: NumberSequence, scale_vectors: bool = True, **kwargs):
        super().__init__(**kwargs)
        "minkowski batch is treated differently for pos or coords rotation"
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.scale_vectors = scale_vectors

    def _apply(self, data: typing.Dict[str, torch.Tensor]):
        scale = random.uniform(*self.scales)
        data["pos"][:, 1:] = data["pos"][:, 1:] * scale
        if self.scale_vectors and "normal" in data:
            raise NotImplementedError("normals have some special consideration maybe. see deltaconv for weirdness")
        if self.scale_vectors and "offset" in data:
            data["offset"] = data["offset"] * scale
        return data

    def __repr__(self):
        return f"{super().__repr__()}({self.scales})"

class RandomTranslateGlobal(Transform):
    """
    Translates shapes by randomly sampled translation values
    within a given interval. This translation happens for the entire shape,
    retaining local relationships.
    Best as a batch transform on the gpu.

    :param translate: Maximum translation in each dimension, defining the
    range (-translate, +translate) per axis to sample from. If a number
    instead of a sequence, the same range is used for each dimension.
    """
    def __init__(self, translate: typing.Union[Number, NumberSequence], **kwargs):
        super().__init__(**kwargs)
        self.translate = translate

    def _apply(self, data: typing.Dict[str, torch.Tensor]):
        euc_dim = data["pos"].shape[-1]
        translate = self.translate
        if isinstance(translate, (int, float)):
            translate = list(itertools.repeat(translate, times = euc_dim))
        assert len(translate) == euc_dim
        transform = []
        for dim in range(euc_dim):
            transform.append(data["pos"].new_empty(1).uniform_(-abs(translate[dim]), abs(translate[dim])))
        data["pos"] = data["pos"] + torch.stack(transform, dim = -1)
        return data

    def __repr__(self):
        return f"{super().__repr__()}({self.translate})"
