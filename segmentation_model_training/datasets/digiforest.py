from pathlib import Path

import numpy as np
import torch
from ..utils.cloud import Cloud

from .base_dataset import BaseDataset

from enum import Enum

from typing import Union

class Labels:
    class From(Enum):
        UNLABELLED = 0
        OUTLIER = 1
        GROUND = 2
        TREE = 3
        SHRUB = 4
        AUTO_GROUND = 98
        OTHER = 99

    class SubFrom(Enum):
        STEM = 1
        CANOPY = 2

    class To(Enum):
        IGNORE = 0
        GROUND = 1
        SHRUB = 2
        STEM = 3
        CANOPY = 4

    def __init__(self, semantics: np.ndarray, instance: np.ndarray) -> None:
        self.semantics = semantics
        self.instance = instance

    @classmethod
    def read(cls, label_fp: Path):
        data = np.fromfile(label_fp, dtype=np.uint32)

        # bits 0-7 correspond to semantic class
        semantic_labels = data & 0xFF
        # bits 8-15 correspond to sub semantic labels -> stem canopy
        subsemantic_labels = (data >> 8) & 0xFF
        # remap everything into just semantics
        semantic_labels = cls.remap(semantic_labels, subsemantic_labels)
        # bits 16-31 correspond to instance
        instance_labels = data >> 16
        return cls(semantic_labels, instance_labels)

    @staticmethod
    def remap(sem_labels: np.ndarray, sub_labels: np.ndarray):
        remapped_labels = sem_labels.copy()
        # tree is handled seperately
        from_to = {
            Labels.From.UNLABELLED.value: Labels.To.IGNORE.value,
            Labels.From.OUTLIER.value: Labels.To.IGNORE.value,
            Labels.From.GROUND.value: Labels.To.GROUND.value,
            Labels.From.SHRUB.value: Labels.To.SHRUB.value,
            Labels.From.AUTO_GROUND.value: Labels.To.GROUND.value,
            Labels.From.OTHER.value: Labels.To.IGNORE.value,
        }
        for key, value in from_to.items():
            remapped_labels[remapped_labels == key] = value

        # handling tree
        tree_mask = sem_labels == Labels.From.TREE.value
        stem_mask = np.logical_and(tree_mask, sub_labels == Labels.SubFrom.STEM.value)
        canopy_mask = np.logical_and(
            tree_mask, sub_labels == Labels.SubFrom.CANOPY.value
        )
        remapped_labels[stem_mask] = Labels.To.STEM.value
        remapped_labels[canopy_mask] = Labels.To.CANOPY.value
        # Tree semantic class doesn't exist anymore. so any remaining tree points, should be none, should be ignored
        stem_and_canopy_mask = np.logical_or(stem_mask, canopy_mask)
        tree_but_not_sub_class_mask = np.logical_and(
            tree_mask, np.logical_not(stem_and_canopy_mask)
        )
        if np.any(tree_but_not_sub_class_mask):
            print("there were remaining unmapped tree values. these are being ignored")
            remapped_labels[tree_but_not_sub_class_mask] = Labels.To.IGNORE.value

        return remapped_labels
    
def voxel_down_sample(
    pos: np.ndarray,
    voxel_size: float,
    intensity: np.ndarray | None = None,
    semantics: np.ndarray | None = None,
    instance: np.ndarray | None = None,
):
    voxel_coords = np.floor(pos / voxel_size).astype(np.int64)

    _, first_indices, inverse = np.unique(
        voxel_coords,
        axis=0,
        return_index=True,
        return_inverse=True,
    )

    # representative point per voxel
    indices = np.sort(first_indices)

    pos_down = pos[indices]

    intensity_down = (
        intensity[indices] if intensity is not None else None
    )

    instance_down = (
        instance[indices] if instance is not None else None
    )

    if semantics is not None:
        semantics_down = np.empty(len(indices), dtype=semantics.dtype)

        for voxel_idx in range(len(indices)):
            voxel_labels = semantics[inverse == voxel_idx]
            semantics_down[voxel_idx] = np.bincount(voxel_labels).argmax()
    else:
        semantics_down = None

    return pos_down, intensity_down, semantics_down, instance_down

class DigiForestDataset(BaseDataset):
    num_classes = len(Labels.To)

    def __init__(
        self,
        root: Path,
        split="trainval",
        mode: str = "scan",
        vds: Union[int, None] = None,
        include_semantics: bool = True,
        include_instance: bool = False,
        include_offsets: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        delete_existing_processed_files: bool = False,
        voxel_size: Union[float, None] = None,
        root_to_data_dir_callable=None,
        return_filename: bool = False,
    ):
        # doktor, reinitialize the state machine
        self.mode = mode
        self.include_semantics = include_semantics
        self.include_instance = include_instance
        self.include_offsets = include_offsets
        self.voxel_size = voxel_size

        self.vds = vds

        # rtdd callable needs to be overridable when using pyg dataset wrapper
        # supremely ugly. but i need something working
        root_to_data_dir_callable = (
            root_to_data_dir_callable or self.default_rtdd_callable
        )

        super().__init__(
            root=root,
            split=split,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            delete_existing_processed_files=delete_existing_processed_files,
            root_to_data_dir_callable=root_to_data_dir_callable,
            return_filename=return_filename,
        )

    def default_rtdd_callable(self, prefix: str, in_path: Path):
        "prefix is either raw or processed"
        if self.vds is not None:
            return in_path / prefix / f"vds{self.vds}"
        else:
            return in_path / prefix

    def _process_raw_file(self, raw_fp: Path):
        """
        process a raw file at raw_fp and save to processed_fp.
        return the data dict.
        also call pre-transform and pre-filter at the end.

        no other logic here
        """
        cloud = Cloud.load(raw_fp)
        label_fp = ((raw_fp / "../../labels").resolve() / raw_fp.name).with_suffix(
            ".label"
        )
        assert label_fp.exists(), f"{label_fp} does not exist for raw_fp"
        labels = Labels.read(label_fp)

        pos = cloud.points.astype(np.float32)  # astype copies if copy!=False
        # sanity checks on data cant happen here? ex. num of points too low etc.
        
        # intensity
        intensity = cloud.get_attribute("intensity").numpy().astype(np.float32)

        # data gets modified based on the state machine
        semantics = None
        if self.include_semantics:
            # int32 causes issues
            semantics = labels.semantics.astype(np.int64).reshape(-1)
            # semantics = semantics - np.min(semantics) # no need to zero after remapping
            if semantics.shape[0] != pos.shape[0]:
                raise ValueError(
                    f"semantics shape {semantics.shape}, pos shape {pos.shape},"
                    " unequal points"
                )

        instance_v = None
        if self.include_instance:
            # instance is int64 (long). int32 causes issues
            instance_v = labels.instance.astype(np.int64).reshape(-1)
            instance_v = instance_v - np.min(instance_v)
            if instance_v.shape[0] != pos.shape[0]:
                raise ValueError(
                    f"instance_v shape {instance_v.shape}, pos shape {pos.shape},"
                    " unequal points"
                )
            
        if self.voxel_size is not None:
            pos, intensity, semantics, instance_v = voxel_down_sample(
                pos,
                self.voxel_size,
                intensity,
                semantics,
                instance_v,
            )
            
        pos_t = torch.from_numpy(pos)
        data = {"pos": pos_t}
        intensity_t = torch.from_numpy(intensity)
        data["intensity"] = intensity_t

        if semantics is not None:
            semantics_t = torch.from_numpy(semantics)
            data["semantics"] = semantics_t

        if instance_v is not None:
            instance_t = torch.from_numpy(instance_v)
            data["instance"] = instance_t
            # useful meta information
            # data["num_instances"] = instance_t.unique().shape[0]

        if self.include_offsets:
            raise NotImplementedError

        if self.pre_filter is not None and self.pre_filter(data):
            # pre_filter returns True, means sample should be filtered
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data
