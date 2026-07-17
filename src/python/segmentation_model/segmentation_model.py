import lightning.pytorch as pl
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from .classwise_wrapper import ClasswiseWrapper
from torch import nn
from .ptv3 import PointTransformerV3
import numpy as np

# from softgroup
class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, batch_norm=True, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if batch_norm:
                modules.append(nn.BatchNorm1d(in_channels, eps=1e-4, momentum=0.1))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        super().__init__(*modules)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)

class PointTransformerV3Lightning(pl.LightningModule):
    def __init__(
        self,
        vds: float = 0.1,
        in_channels=1,
        embedding_size=8,
        num_classes=5,
        coord_dimension=3,
        lr: float = 0.001,
        batch_size: int = 1,
        clear_torch_cache=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.setup_metrics()

        self.point_transformer = PointTransformerV3(
            in_channels=self.hparams.in_channels,
            stride=(2, 2, 2),
            enc_depths=(1, 1, 1, 1),
            enc_channels=(embedding_size, embedding_size*2, embedding_size*4, embedding_size*8),
            enc_num_head=(1, 1, 2, 4),
            enc_patch_size=(256, 256, 256, 256),
            dec_depths=(1, 1, 1),
            dec_channels=(embedding_size*4, embedding_size*2, embedding_size),
            dec_num_head=(2, 1, 1),
            dec_patch_size=(256, 256, 256),
            mlp_ratio=2,
        )

        self.segmentation_head = MLP(
            embedding_size*4,
            self.hparams.num_classes,
            batch_norm=True,
            num_layers=2,
        )

        self.loss = torch.nn.CrossEntropyLoss(
            reduction="mean", ignore_index=0
        )

    def setup_metrics(self):
        metrics = MetricCollection(
            {
                "Mean_IOU": MulticlassJaccardIndex(
                    num_classes=self.hparams.num_classes,
                    # TODO: hardcoded ignore index, same in loss
                    ignore_index=0,
                    average="macro",
                ),
                "Mean_F1": MulticlassF1Score(
                    num_classes=self.hparams.num_classes,
                    ignore_index=0,
                    average="macro",
                ),
                "IOU": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=self.hparams.num_classes,
                        ignore_index=0,
                        average="none",
                    ),
                    labels=["Ignore", "Ground", "Shrub", "Stem", "Canopy"],
                ),
            },
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, batch):
        """
        batch should be a dict with keys pos, intensity.
        coords (pos) should be batched with shape Nx4, where the first column is batch_idx
        """
        coord = batch["pos"][:, 1:]
        data_dict = {
            "feat": torch.ones((coord.shape[0], 1), device = coord.device, dtype = coord.dtype),
            "coord": coord,
            "batch": batch["pos"][:, 0].long(),
            "grid_size": 0.1
        }
        
        # forward through PTv3
        predicted = self.point_transformer(data_dict)    

        # segmentation head
        logits = self.segmentation_head(predicted.feat)
        return logits
    
    def infer(self, batch):
        logits = self.forward(batch)
        conf = torch.softmax(logits, dim=-1)
        pred = torch.argmax(conf, dim=-1)
        return pred, conf

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["semantics"]
        logits = self.forward(batch)
        loss = self.loss(logits, labels.long())
        self.log(
            "train/loss",
            loss,
            batch_size=self.hparams.batch_size,
            on_step=True,
            on_epoch=False,
        )
        self.train_metrics.update(logits, labels.int())
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        labels = batch["semantics"]
        logits = self.forward(batch)
        loss = self.loss(logits, labels.long())

        self.log(
            "val/loss",
            loss,
            batch_size=self.hparams.batch_size,
            on_step=True,
            on_epoch=False,
        )
        # see training_step for some comments
        self.val_metrics.update(logits, labels.int())
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """same as validation step. only difference is its called when .test() is explicitly called on the trainer"""
        labels = batch["semantics"]
        logits = self.forward(batch)
        self.test_metrics.update(logits, labels.int())
        return logits

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        logits = self.forward(batch)
        pred = logits.max(dim=1)[1]
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=False,  # might be useful for small dataset
        )
        # can add a lr scheduler
        return {"optimizer": optimizer}

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.hparams.clear_torch_cache:
            # this is horrible. but theres no other way to train on my limited memory gpu
            torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        # we need to manually log at the end of epoch level metrics because we use classwisewrapper
        # see https://github.com/Lightning-AI/torchmetrics/issues/2091
        self.log_dict(
            self.train_metrics.compute(),
            batch_size=self.hparams.batch_size,
        )
        self.train_metrics.reset()

    def on_validation_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.hparams.clear_torch_cache:
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(
            self.val_metrics.compute(),
            batch_size=self.hparams.batch_size,
        )
        self.val_metrics.reset()

    def on_test_batch_end(
        self, outputs, batch, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.hparams.clear_torch_cache:
            torch.cuda.empty_cache()

    def on_test_epoch_end(self) -> None:
        self.log_dict(
            self.test_metrics.compute(),
            batch_size=self.hparams.batch_size,
        )
        self.test_metrics.reset()
