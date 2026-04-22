from pathlib import Path
import copy
from typing import Callable, Union
import torch
from ..utils.io import load_yaml_or_json
from torch.utils.data import Dataset
from functools import cached_property

from typing import Dict as dict
from typing import List as list
from typing import Tuple as tuple

from ..utils.serialize import TorchSerializedList


class BaseDataset(Dataset):
    """
    A base dataset class that takes inspiration from pytorch
    geometric's dataset class. In essence all that should be needed is to implement
    _process_raw_file and have a set structure to the dataset folder layout. raw data
    should be inside "root/raw". processed data (from _process_raw_file) ends up in
    "root/processed". in case some subdir fuckery is needed, for example files are voxel
    downsampled with different sizes and stored in individual vds subdirs, the
    root_to_data_dir_callable can be passed which can handle arbitrary modifications to
    the raw_dir and processed_dir properties. Data split into different parts is handled
    by the self.data_split property. has a hardcoded requirement of the file
    root/raw/data_split.json. this json should be a dict with keys for the data
    categories, i.e., train, test, trainval, etc., and the values are lists of file paths relative
    to the original root/raw data directory without any subdir fuckery, thats handled
    by the callable, so ideally the list values are just file names.
    if you want to load a combined trainval split, that means the json should have a
    trainval key and the list should include file names corresponding to both train and
    val. no sort of merging category logic is implemented here.
    """

    # can be useful to ensure checks that model output channels equal num classes
    num_classes = 0

    def __init__(
        self,
        root: Path,
        split="train",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        delete_existing_processed_files: bool = True,
        root_to_data_dir_callable: Union[Callable[[str, Path], Path], None] = None,
        return_filename: bool = False,
    ):
        """
        :param return_filename: modifies getitem to return an additional str value. this should not be necessary during training
        """
        super().__init__()
        # doktor, initialize the state machine
        self.root = root
        self.split = split
        if split not in self.data_split.keys():
            raise Exception(
                f'split {split} not present in the data split json at {root / "raw" / "data_split.json"}'
            )

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.delete_existing_processed_files = delete_existing_processed_files
        self.root_to_data_dir_callable = root_to_data_dir_callable
        self.return_filename = return_filename
        self._ensure_processed_data_existence()

        self.sample_fps = TorchSerializedList(self.get_sample_fps())

    def default_rtdd_callable(self, prefix: str, in_path: Path):
        "prefix is either raw or processed"
        raise NotImplementedError

    def __len__(self):
        return len(self.sample_fps)

    def __getitem__(self, idx):
        sample_fp: Path = self.sample_fps[idx]
        print(f"loading {sample_fp}")
        data: dict[str, Union[str, torch.Tensor]] = torch.load(sample_fp)
        if self.transform is not None:
            data = self.transform(data)
        if self.return_filename:
            data["filename"] = sample_fp.relative_to(self.root).as_posix()
        return data

    @cached_property
    def data_split(self) -> dict[str, list[Path]]:
        """hard coded requirement of existence of self.root / raw / data_split.json"""
        file_path = self.root / "raw" / "data_split.json"
        if not file_path.exists():
            raise Exception(f"File {file_path} is required for a dataset.")
        data_split: dict[str, list[Path]] = load_yaml_or_json(
            self.root / "raw" / "data_split.json"
        )
        return data_split

    @property
    def raw_dir(self):
        if self.root_to_data_dir_callable is None:
            return (self.root / "raw").resolve()
        else:
            return self.root_to_data_dir_callable("raw", self.root)

    @property
    def processed_dir(self):
        if self.root_to_data_dir_callable is None:
            return (self.root / "processed").resolve()
        else:
            return self.root_to_data_dir_callable("processed", self.root)

    def get_sample_fps(self) -> list[Path]:
        """gets the files from processed dir as per the state machine"""
        # its actually list[str], but whatever to satisfy mypy
        fp_list: list[Path] = self.data_split[self.split]
        # sanity checking
        assert isinstance(fp_list, list), fp_list
        # fp list is the raw file list. need to convert to processed file loc list
        processed_fp_list = copy.deepcopy(fp_list)
        for idx in range(len(fp_list)):
            processed_fp_list[idx] = (self.processed_dir / fp_list[idx]).with_suffix(
                ".pt"
            )
            assert processed_fp_list[
                idx
            ].exists(), f"processed file {processed_fp_list[idx]} doesn't exist"
        return processed_fp_list

    def _ensure_processed_data_existence(self):
        """
        the raw file paths should resolve.
        should the processed files not exist, this runs any pre-processing
        logic, i.e. pre-transforms and pre-filters, on the raw data and stores in
        processed dirs.
        """
        # TODO: somehow dump the dataset configuration as a pickle. and compare that too
        # so that if the config changes, data is reprocessed
        if self.pre_filter is not None:
            raise NotImplementedError(
                "this needs additional logic and a partial rewrite since data_split needs to be updated and stored inside the processed dir."
            )
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        fp_list = copy.deepcopy(self.data_split[self.split])

        for fp_name in fp_list:
            # for fp_name in bar(
            #     fp_list, desc=f"Checking files for {self.split}", total=len(fp_list)
            # ):
            raw_fp: Path = self.raw_dir / fp_name
            assert (
                raw_fp.exists() and raw_fp.is_file()
            ), f"{raw_fp.exists()} and {raw_fp.is_file()}, raw_fp {raw_fp}"
            processed_fp = (self.processed_dir / fp_name).with_suffix(".pt")
            if processed_fp.exists():
                if not self.delete_existing_processed_files:
                    continue
                else:
                    print(f"removing existing file at {processed_fp}")
                    processed_fp.unlink()
            # TODO: pre_filter can return None, in which case this sample should be removed
            # from processed data_split.
            processed_fp.parent.mkdir(parents=True, exist_ok=True)
            data = self._process_raw_file(raw_fp)
            torch.save(data, processed_fp)
            print(f"raw file - {raw_fp} - processed to - {processed_fp}")

    def _process_raw_file(self, raw_fp: Path):
        """
        process a raw file at raw_fp and save to processed_fp.
        processed_fp will be a file ending with .pt.
        processed data should be a dict(like) with pos, labels, instance etc tensors
        with respective str keys.
        also call pre-transform and pre-filter at the end.
        return data

        no other logic here
        """
        raise NotImplementedError
