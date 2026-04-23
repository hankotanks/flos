from pathlib import Path
from typing import Callable, Type, Dict, List, Union

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from ..datasets.base_dataset import BaseDataset


class BaseDataModule(pl.LightningDataModule):
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
        dataset_cls: Union[Type[BaseDataset], None] = None,
        prepare_splits: List[str] = ["train", "val", "test", "predict"],
        collate_fn=None,
    ):
        """
        transform and batch_transform are not applied on validation data
        NOTE: if you want the transforms (augmentations) calculated on gpu
        use batch_transform instead of tranform. transform will likely
        be on the cpu, because that should happen before the
        transfer_batch_to_device() lightning hook. unless you transfer to
        device while in __getitem__ of the dataset. meanwhile
        on_after_batch_transfer() is explicitly for ops to be run after
        transfer happens. see
        https://pytorch-lightning.readthedocs.io/en/2.0.1.post0/common/lightning_module.html#hooks
        and batch_transform is what is called in this class's hook here
        """
        super().__init__()
        # calling save_hyperparams gives you every init arg as part of the self.hparams dict
        # TODO. im curious how the transforms will be dumped. probably need proper reprs
        # this also necessitates # pyright: ignore all over the place, which is super
        # ugly. need to somehow declare that hparams has those keys and thos types.
        self.save_hyperparameters(
            ignore=[
                "data_dir",
                "transform",
                "batch_transform",
                "pre_transform",
                "pre_filter",
                "dataset_cls",
                "prepare_splits",
                "collate_fn",
            ]
        )
        # store the ignored as attributes
        self.data_dir = data_dir
        self.transform = transform
        self.batch_transform = batch_transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.dataset_cls = dataset_cls if dataset_cls is not None else BaseDataset
        self.prepare_splits = prepare_splits
        self.collate_fn = collate_fn

    def prepare_data(self):
        """
        download and save data.
        one time processes are here. always called from the main worker
        when multiple processes are used
        do no assign state here, do it in setup or init
        """

        def summarize_dataset(name: str, dataset):
            print("=" * 10)
            print(f"{name} dataset has {len(dataset)} samples")
            print("=" * 10)

        # ensure data state. init-ing below, calls the preprocess stuff
        for split in self.prepare_splits:
            summarize_dataset(
                split,
                self.dataset_cls(
                    root=self.data_dir,
                    split=split,
                    transform=None,
                    pre_transform=self.pre_transform,
                    pre_filter=self.pre_filter,
                    **self.hparams.dataset_config,  # pyright: ignore
                ),
            )

    def setup(self, stage=None):
        # Create datasets and set state
        if stage == "fit":
            self.train_data = self.dataset_cls(
                root=self.data_dir,
                split="train",
                transform=self.transform,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )
        if stage == "fit" or stage == "validate":
            self.val_data = self.dataset_cls(
                root=self.data_dir,
                split="val",
                transform=None,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )
        if stage == "test":
            self.test_data = self.dataset_cls(
                root=self.data_dir,
                split="test",
                transform=None,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )
        if stage == "predict":
            self.predict_data = self.dataset_cls(
                root=self.data_dir,
                split="predict",
                transform=None,
                pre_transform=self.pre_transform,
                pre_filter=self.pre_filter,
                **self.hparams.dataset_config,  # pyright: ignore
            )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        return loader

    def predict_dataloader(self):
        loader = DataLoader(
            self.predict_data,
            batch_size=self.hparams.batch_size,  # pyright: ignore
            num_workers=self.hparams.num_workers,  # pyright: ignore
            shuffle=False,
            drop_last=False,
            collate_fn=self.collate_fn,
        )
        return loader

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """the batch will have been transferred to device when this is called."""
        # if self.trainer.training and self.batch_transform:  # pyright: ignore
        if self.batch_transform:
            batch = self.batch_transform(batch)
        return batch
