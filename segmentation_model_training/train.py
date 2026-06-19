import copy
import os
from pathlib import Path
from typing import Optional, Dict, List

import segmentation_model_training.utils.transforms as fAIT
import torch
import typer
from segmentation_model_training.data_modules.digiforest import DigiForestDataModule
from src.python.segmentation_model.segmentation_model import PointTransformerV3Lightning
from segmentation_model_training.utils.config import sync_config_keys
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, Logger
from segmentation_model_training.utils.io import load_yaml_or_json
from torchvision.transforms import Compose

cli = typer.Typer()

def train(
    model_config: Dict,
    lightning_config: Dict,
    datamodule_config: Dict,
    log_dir: Path,
    data_dir: Path,
    logger: List[Logger],
    seed=None,
    verbose=False
):
    # Data Prep
    # ----------------------
    # TODO: log all the transforms as well
    pre_transfom = Compose([fAIT.CenterGlobal(verbose=verbose)])
    batch_transform = Compose([
        fAIT.RandomRotate(180, axis=2, verbose=verbose),
        fAIT.RandomUniformScale((0.8, 1.2), verbose=verbose),
    ])
    dataset_config = datamodule_config.pop("dataset")
    if "debug_no_augmentation" in datamodule_config:
        debug_no_augmentation = datamodule_config.pop("debug_no_augmentation")
        batch_transform = None if debug_no_augmentation else batch_transform
    datamodule = DigiForestDataModule(
        data_dir=data_dir,
        dataset_config=dataset_config,
        transform=None,
        batch_transform=batch_transform,
        pre_transform=pre_transfom,
        pre_filter=None,
        **datamodule_config,
    )
    # ----------------------

    # Model
    # ----------------------
    model = PointTransformerV3Lightning(**model_config)
    # sanity check model and dataset compatibility
    assert (
        model.hparams.num_classes == datamodule.dataset_cls.num_classes
    ), f"model num_classes {model.hparams.num_classes} doesnt match the dataset {datamodule.dataset_cls.num_classes}"

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # metric_key = "val/MulticlassAccuracy"
    metric_key = lightning_config.pop("checkpoint_metric_key")
    f1_ckpt = ModelCheckpoint(
        dirpath=None,  # use the default_root_dir of trainer
        monitor=metric_key,
        filename=metric_key.replace("/", "_") + "_{epoch:02d}_{" + metric_key + ":.2f}",
        auto_insert_metric_name=False,
        mode="max",
        save_last=True,
        verbose=verbose,
    )
    # ----------------------

    # Train
    # ----------------------
    # using a CUDA device ('NVIDIA RTX A5000') that has Tensor Cores. To properly utilize them, set
    # `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance.
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")
    del lightning_config["seed"]
    # profiler = PyTorchProfiler(profile_memory=True)
    trainer = Trainer(
        logger=logger,
        callbacks=[lr_monitor, f1_ckpt],
        default_root_dir=log_dir,
        **lightning_config,
        # resume_from_checkpoint=ckpt,
        # profiler=profiler,
    )
    trainer.fit(model, datamodule=datamodule)
    # ----------------------

    # Test
    # ----------------------
    if "test" in datamodule.prepare_splits:
        trainer.test(model, datamodule=datamodule)


@cli.command()
def main(
    model_config_fp: Path = typer.Option(
        ...,
        "--model-conf",
        help="path to model configuration file",
    ),
    lightning_config_fp: Path = typer.Option(
        ...,
        "--lightning-conf",
        help="path to lightning configuration file",
    ),
    data_config_fp: Path = typer.Option(
        ...,
        "--data-conf",
        help="path to data module configuration file",
    ),
    log_dir: Path = typer.Option(
        ..., help="Root directory for log_dir/experiment_name/run_name"
    ),
    data_dir: Path = typer.Option(..., help="Directory of data. should have ./raw"),
    experiment_name: str = typer.Option(
        ..., help="The name of the experiment, e.g. kpconv"
    ),
    run_name: Optional[str] = typer.Option(
        None, help="The name of a run. default will be version_#"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable debug: clean run dir, activate lightning debug args, and perhaps other stuff",
    ),
    threads: Optional[int] = typer.Option(
        None, help="Number of threads/workers to use for the datamodule"
    ),
):
    # catch the cli args first without modification
    cli_args = copy.deepcopy(locals())
    cli_args["execution_dir"] = os.getcwd()
    # first up
    log_dir.mkdir(parents=True, exist_ok=True)
    # set up the tb logger since that handles creation of the run dir,
    # in case no name was passed
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        default_hp_metric=False,
        version=run_name,
    )
    # get the actual run log dir
    run_dir = Path(tb_logger.log_dir)
    if debug:
        # remove the run dir, since we dont want to bloat the runs
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    cli_args["run_dir"] = run_dir

    # Config
    # ----------------------
    # load confs
    model_config = load_yaml_or_json(model_config_fp)
    datamodule_config = load_yaml_or_json(data_config_fp)
    lightning_config: Dict = load_yaml_or_json(lightning_config_fp)

    # process and overwrite confs
    if "debug" in lightning_config:
        # debug sub dicts of config should be put at top level
        if debug:
            lightning_config.update(lightning_config["debug"])
        del lightning_config["debug"]
    if "debug" in model_config:
        if debug and model_config["debug"] is not None:
            model_config.update(model_config["debug"])
        del model_config["debug"]
    if threads is not None:
        if threads != datamodule_config["num_workers"]:
            print(
                f"WARN: Datamodule config num_workers = {datamodule_config['num_workers']}, overwriting with cli arg {threads}"
            )
        datamodule_config["num_workers"] = threads
    # ensure good configuration
    sync_config_keys(
        [cli_args, lightning_config, model_config, datamodule_config], ["batch_size"]
    )

    # ----------------------

    # sets seeds for numpy, torch and python.random.
    seed_everything(lightning_config["seed"], workers=True)
    # just in case
    data_dir = data_dir.resolve()

    train(
        model_config=model_config,
        lightning_config=lightning_config,
        datamodule_config=datamodule_config,
        log_dir=run_dir,
        data_dir=data_dir,
        logger=[tb_logger],
        seed=lightning_config["seed"],
        verbose=debug
    )


if __name__ == "__main__":
    cli()
