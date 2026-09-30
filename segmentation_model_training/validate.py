import pathlib
import typing
import argparse

import torch
import torchvision.transforms as transforms

import lightning.pytorch as lightning

import segmentation_model_training.utils as utils
import segmentation_model_training.utils.config
import segmentation_model_training.utils.io
import segmentation_model_training.utils.transforms
import segmentation_model_training.datamodules as datamodules
import segmentation_model_training.datamodules.digiforest

import src.python.segmentation_model.segmentation_model as segmentation_model

TRAIN_ROOT = pathlib.Path(__file__).resolve(strict = True).parent
MODEL_ROOT = (TRAIN_ROOT.parent / 'models').resolve(strict = True)

def validate(config_model: typing.Dict, config_trainer: typing.Dict, config_data: typing.Dict, 
    ckpt: pathlib.Path, data: pathlib.Path, verbose = False):
    # data prep
    transform_pre = transforms.Compose([utils.transforms.CenterGlobal(verbose = verbose)])
    config_dataset = config_data.pop("dataset")
    if "debug_no_augmentation" in config_data:
        config_data.pop("debug_no_augmentation")
    datamodule = datamodules.digiforest.DigiForestDataModule(data_dir = data, dataset_config = config_dataset, transform = None,
        batch_transform = None, pre_transform = transform_pre, pre_filter = None, **config_data)
    
    # model
    model = segmentation_model.PointTransformerV3Lightning(**config_model)
    # sanity check model and dataset compatibility
    assert model.hparams.num_classes == datamodule.dataset_cls.num_classes, \
        f"model num_classes {model.hparams.num_classes} doesnt match the dataset {datamodule.dataset_cls.num_classes}"

    # validation
    torch.set_float32_matmul_precision("medium")
    del config_trainer["seed"]
    if "checkpoint_metric_key" in config_trainer:
        config_trainer.pop("checkpoint_metric_key")
    trainer = lightning.Trainer(logger = False, **config_trainer) # resume_from_checkpoint=ckpt, # profiler=profiler,
    trainer.validate(model = model, datamodule = datamodule, ckpt_path = str(ckpt))

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = pathlib.Path, required = True,
        help = "path to data (containing ./raw, etc)")
    parser.add_argument("--ckpt", type = pathlib.Path, required = True, 
        help = "path to the checkpoint")
    parser.add_argument("--threads", type = int, default = None,
        help = "number of threads to use for the data module")
    parser.add_argument("--debug", action = "store_true",
        help = "enable debug")
    return parser.parse_args()

def main(args: argparse.Namespace): 
    # load model config
    path_config_model = pathlib.Path(TRAIN_ROOT / 'config' / 'model.yaml').resolve(strict = True)
    config_model: typing.Dict = utils.io.load_yaml(path_config_model)
    # load data module config
    path_config_data = pathlib.Path(TRAIN_ROOT / 'config' / 'data.yaml').resolve(strict = True)
    config_data: typing.Dict = utils.io.load_yaml(path_config_data)
    # load trainer config
    path_config_trainer = pathlib.Path(TRAIN_ROOT / 'config' / 'trainer.yaml').resolve(strict = True)
    config_trainer: typing.Dict = utils.io.load_yaml(path_config_trainer)

    if "debug" in config_trainer:
        if args.debug:
            config_trainer.update(config_trainer["debug"])
        del config_trainer["debug"]
    if "debug" in config_model:
        if args.debug and config_model["debug"] is not None:
            config_model.update(config_model["debug"])
        del config_model["debug"]
    if args.threads is not None:
        if args.threads != config_data["num_workers"]:
            print(f"WARN: Datamodule config num_workers = {config_data['num_workers']}, overwriting with cli arg {args.threads}")
        config_data["num_workers"] = args.threads
    # ensure good configuration
    utils.config.sync_config_keys([config_trainer, config_model, config_data], ["batch_size"])

    # sets seeds for numpy, torch and python.random.
    lightning.seed_everything(config_trainer["seed"], workers = True)

    # setup training
    validate(config_model = config_model, config_trainer = config_trainer, config_data = config_data,
        ckpt = args.ckpt.resolve(strict = True), data = args.data.resolve(strict = True), verbose = args.debug)

if __name__ == "__main__":
    main(args())
