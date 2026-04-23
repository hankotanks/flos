# proof of concept

from pathlib import Path
from typing import Dict, List, Callable, Union
import torch

from ..datasets.digiforest import DigiForestDataset
from .base_data_module import BaseDataModule

def collate_fn(batch: List[Dict[str, torch.Tensor]]):
    """
    by default, with how we do our dataloaders, they return a dict of key value tensors.
    the batch here will be a list of such dicts.
    models expect a single dict with original keys and batched tensors.
    """
    batch_dict_values = [data_dict.values() for data_dict in batch]
    # this assumes that the order of keys in the dict is the following
    # i can make it general. but i cba.
    pos_list, intensity_list, semantics_list, _ = list(zip(*batch_dict_values))
    # dict_keys(['pos', 'intensity', 'semantics', 'instance'])

    batched_pos_list = []
    for i, pos in enumerate(pos_list):
        batch_idx = i * torch.ones_like(pos[:, 0])
        batched_pos = torch.hstack((batch_idx.reshape(-1, 1), pos))
        batched_pos_list.append(batched_pos)
    batched_pos = torch.cat(batched_pos_list, dim=0)
    batched_intensity = torch.cat(intensity_list, dim=0)
    batched_semantics = torch.cat(semantics_list, dim=0)
    return {
        "pos": batched_pos,
        "intensity": batched_intensity,
        "semantics": batched_semantics,
    }


class DigiForestDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        num_workers: int,
        dataset_config: Dict,
        transform: Union[Callable[..., Dict], None] = None,
        batch_transform: Union[Callable[..., Dict], None] = None,
        pre_transform: Union[Callable[..., Dict], None] = None,
        pre_filter: Union[Callable[..., bool], None] = None,
        prepare_splits: List[str] = ["train", "val"],
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_config=dataset_config,
            transform=transform,
            batch_transform=batch_transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            prepare_splits=prepare_splits,
            dataset_cls=DigiForestDataset,
            collate_fn=collate_fn,
        )
