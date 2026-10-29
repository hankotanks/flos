import pathlib
import enum
import typing

import numpy as np
import torch

from ..utils.cloud import Cloud
from .base import BaseDataset

class Labels:
    class From(enum.Enum):
        UNLABELLED = 0
        OUTLIER = 1
        GROUND = 2
        TREE = 3
        SHRUB = 4
        AUTO_GROUND = 98
        OTHER = 99

    class SubFrom(enum.Enum):
        STEM = 1
        CANOPY = 2

    class To(enum.Enum):
        IGNORE = 0
        GROUND = 1
        VEGETATION = 2
        STEM = 3

    def __init__(self, semantics: np.ndarray, instance: np.ndarray) -> None:
        self.semantics = semantics
        self.instance = instance

    @classmethod
    def read(cls, label_fp: pathlib.Path):
        data = np.fromfile(label_fp, dtype = np.uint32)

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
            Labels.From.SHRUB.value: Labels.To.VEGETATION.value,
            Labels.From.AUTO_GROUND.value: Labels.To.GROUND.value,
            Labels.From.OTHER.value: Labels.To.IGNORE.value,
        }
        for key, value in from_to.items():
            remapped_labels[remapped_labels == key] = value

        # handling tree
        tree_mask = sem_labels == Labels.From.TREE.value
        stem_mask = np.logical_and(tree_mask, sub_labels == Labels.SubFrom.STEM.value)
        canopy_mask = np.logical_and(tree_mask, sub_labels == Labels.SubFrom.CANOPY.value)
        remapped_labels[stem_mask] = Labels.To.STEM.value
        remapped_labels[canopy_mask] = Labels.To.VEGETATION.value
        
        # Tree semantic class doesn't exist anymore. so any remaining tree points, should be none, should be ignored
        stem_and_canopy_mask = np.logical_or(stem_mask, canopy_mask)
        tree_but_not_sub_class_mask = np.logical_and(tree_mask, np.logical_not(stem_and_canopy_mask))
        if np.any(tree_but_not_sub_class_mask):
            print("there were remaining unmapped tree values. these are being ignored")
            remapped_labels[tree_but_not_sub_class_mask] = Labels.To.IGNORE.value

        return remapped_labels
    
# def print_label_distribution(labels, name = ""):
#     counts = np.bincount(labels, minlength = len(Labels.To))
#     proportions = counts / len(labels)
# 
#     print(f"{name} ({len(labels)} points):")
#     for idx, (count, prop) in enumerate(zip(counts, proportions)):
#         print(f"  {Labels.To(idx).name:>8}: {count:>8} ({prop * 100:6.2f}%)")

# CLASS_WEIGHTS = np.ones(len(Labels.To), dtype = np.float32)

# threshold for preserving GROUND/STEM
THRESHOLD = 0.5
    
def voxel_down_sample(pos: np.ndarray, voxel_size: float,
    intensity: typing.Union[np.ndarray, None] = None,
    semantics: typing.Union[np.ndarray, None] = None,
    instances: typing.Union[np.ndarray, None] = None):
    voxel_coords = np.floor(pos / voxel_size).astype(np.int64)

    unique_voxels, _, indices_inverse = np.unique(voxel_coords,
        axis = 0, return_index = True, return_inverse = True)

    # reorder representatives to match inverse voxel ids
    representative_indices = np.full(len(unique_voxels), len(pos), dtype = np.int64)
    np.minimum.at(representative_indices, indices_inverse, np.arange(len(pos)))

    num_voxels = len(unique_voxels)

    pos_down = np.zeros((num_voxels, 3), dtype = pos.dtype)
    np.add.at(pos_down, indices_inverse, pos)
    voxel_counts = np.bincount(indices_inverse)
    pos_down /= voxel_counts[:, None]

    intensity_down = intensity[representative_indices] if intensity is not None else None

    if semantics is not None:
        semantics_down = np.full(num_voxels, Labels.To.IGNORE.value, dtype = semantics.dtype)
        valid = semantics != Labels.To.IGNORE.value

        '''
        weighted_counts = np.zeros((num_voxels, len(Labels.To)), dtype = np.float32)
        np.add.at(weighted_counts, (indices_inverse[valid], semantics[valid]), CLASS_WEIGHTS[semantics[valid]])
        labelled = weighted_counts.sum(axis=1) > 0
        semantics_down[labelled] = weighted_counts[labelled].argmax(axis = 1)
        '''

        counts = np.zeros((num_voxels, len(Labels.To)), dtype = np.int32)
        np.add.at(counts, (indices_inverse[valid], semantics[valid]), 1)
        labelled = counts.sum(axis = 1) > 0
        for voxel_idx in np.where(labelled)[0]:
            voxel_counts = counts[voxel_idx]
            order = np.argsort(voxel_counts)
            label_fst = order[-1]
            label_snd = order[-2]
            count_fst = voxel_counts[label_fst]
            count_snd = voxel_counts[label_snd]
            # preserve thin structures and ground when the vote is close.
            if count_snd >= THRESHOLD * count_fst:
                if label_fst == Labels.To.VEGETATION.value and label_snd == Labels.To.STEM.value:
                    label_fst = Labels.To.STEM.value
                elif label_fst == Labels.To.VEGETATION.value and label_snd == Labels.To.GROUND.value:
                    label_fst = Labels.To.GROUND.value
            semantics_down[voxel_idx] = label_fst
    else:
        semantics_down = None

    if instances is not None:
        instances_down = np.empty(num_voxels, dtype = instances.dtype)
        # preserve correct order
        order = np.argsort(indices_inverse)
        sorted_inverse = indices_inverse[order]
        sorted_instance = instances[order]
        # majority voting for instances
        voxel_starts = np.r_[0, np.where(np.diff(sorted_inverse))[0] + 1]
        voxel_ends = np.r_[voxel_starts[1:], len(order)]
        for voxel_idx, (start, end) in enumerate(zip(voxel_starts, voxel_ends)):
            ids = sorted_instance[start:end]
            values, counts = np.unique(ids, return_counts = True)
            instances_down[voxel_idx] = values[counts.argmax()]
    else:
        instances_down = None

    return pos_down, intensity_down, semantics_down, instances_down,

class DigiForestDataset(BaseDataset):
    num_classes = len(Labels.To)

    def __init__(self,
        root: pathlib.Path,
        split = "trainval",
        mode: str = "scan",
        vds: typing.Union[int, None] = None,
        include_semantics: bool = True,
        include_instance: bool = False,
        include_offsets: bool = False,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        delete_existing_processed_files: bool = False,
        voxel_size: typing.Union[float, None] = None,
        root_to_data_dir_callable = None,
        return_filename: bool = False):
        # doktor, reinitialize the state machine
        self.mode = mode
        self.include_semantics = include_semantics
        self.include_instance = include_instance
        self.include_offsets = include_offsets
        self.voxel_size = voxel_size

        self.vds = vds

        # rtdd callable needs to be overridable when using pyg dataset wrapper
        # supremely ugly. but i need something working
        root_to_data_dir_callable = (root_to_data_dir_callable or self.default_rtdd_callable)

        super().__init__(
            root = root,
            split = split,
            transform = transform,
            pre_transform = pre_transform,
            pre_filter = pre_filter,
            delete_existing_processed_files = delete_existing_processed_files,
            root_to_data_dir_callable = root_to_data_dir_callable,
            return_filename = return_filename,
        )

    def default_rtdd_callable(self, prefix: str, in_path: pathlib.Path):
        "prefix is either raw or processed"
        if self.vds is not None:
            return in_path / prefix / f"vds{self.vds}"
        else:
            return in_path / prefix

    def _process_raw_file(self, raw_fp: pathlib.Path):
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
                raise ValueError(f"semantics shape {semantics.shape}, pos shape {pos.shape}, unequal points")

        instance_v = None
        if self.include_instance:
            # instance is int64 (long). int32 causes issues
            instance_v = labels.instance.astype(np.int64).reshape(-1)
            instance_v = instance_v - np.min(instance_v)
            if instance_v.shape[0] != pos.shape[0]:
                raise ValueError(f"instance_v shape {instance_v.shape}, pos shape {pos.shape}, unequal points")
            
        # print_label_distribution(semantics, name = "before")
        if self.voxel_size is not None:
            pos, intensity, semantics, instance_v = voxel_down_sample(
                pos,
                self.voxel_size,
                intensity,
                semantics,
                instance_v,
            )

        # print_label_distribution(semantics, name = "after")
            
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
