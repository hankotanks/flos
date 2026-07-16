import pathlib
import typing
import torch
import argparse

import lightning.pytorch as lightning
import torchvision.transforms as transforms

import segmentation_model_training.utils as utils
import segmentation_model_training.utils.config
import segmentation_model_training.utils.io
import segmentation_model_training.utils.transforms
import segmentation_model_training.data_modules as data_modules
import segmentation_model_training.data_modules.digiforest

import src.python.segmentation_model.segmentation_model as segmentation_model

TRAIN_ROOT = pathlib.Path(__file__).resolve(strict = True).parent
MODEL_ROOT = (TRAIN_ROOT.parent / 'models').resolve(strict = True)

def train(config_model: typing.Dict, config_trainer: typing.Dict, config_data: typing.Dict, log: pathlib.Path, data: pathlib.Path, logger: typing.List[lightning.loggers.Logger], seed = None, verbose = False):
    # data prep
    transform_pre = transforms.Compose([utils.transforms.CenterGlobal(verbose = verbose)])
    transform_batch = transforms.Compose([
        utils.transforms.RandomRotate(180, axis = 2, verbose = verbose),
        utils.transforms.RandomUniformScale((0.8, 1.2), verbose = verbose),
    ])
    config_dataset = config_data.pop("dataset")
    if "debug_no_augmentation" in config_data:
        debug_no_augmentation = config_data.pop("debug_no_augmentation")
        transform_batch = None if debug_no_augmentation else transform_batch
    datamodule = data_modules.digiforest.DigiForestDataModule(data_dir = data, dataset_config = config_dataset, transform = None,
        batch_transform = transform_batch, pre_transform = transform_pre, pre_filter = None, **config_data)
    
    # model
    model = segmentation_model.PointTransformerV3Lightning(**config_model)
    # sanity check model and dataset compatibility
    assert model.hparams.num_classes == datamodule.dataset_cls.num_classes, \
        f"model num_classes {model.hparams.num_classes} doesnt match the dataset {datamodule.dataset_cls.num_classes}"

    # learning rate monitor
    monitor = lightning.callbacks.LearningRateMonitor(logging_interval = "step")
    
    # checkpoint
    ckpt_key = config_trainer.pop("checkpoint_metric_key") # metric_key = "val/MulticlassAccuracy"
    ckpt = lightning.callbacks.ModelCheckpoint(
        dirpath = None,  # use the default_root_dir of trainer
        monitor = ckpt_key,
        filename = ckpt_key.replace("/", "_") + "_{epoch:02d}_{" + ckpt_key + ":.2f}",
        auto_insert_metric_name = False,
        mode = "max",
        save_last = True,
        verbose = verbose,
    )
    
    # training
    # using a CUDA device ('NVIDIA RTX A5000') that has Tensor Cores. To properly utilize them, set
    # `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance.
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")
    del config_trainer["seed"]
    # profiler = PyTorchProfiler(profile_memory = True)
    trainer = lightning.Trainer(logger = logger, callbacks = [monitor, ckpt], default_root_dir = log, **config_trainer) # resume_from_checkpoint=ckpt, # profiler=profiler,
    trainer.fit(model, datamodule = datamodule)
   
    # test
    if "test" in datamodule.prepare_splits: trainer.test(model, datamodule = datamodule)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type = pathlib.Path, required = True,
        help = 'path to data (containing ./raw, etc)')
    parser.add_argument("--name", type = str, required = False, 
        default = MODEL_ROOT / str(max([int(p.name) for p in MODEL_ROOT.iterdir() if p.is_dir() and p.name.isdigit()], default = -1) + 1), 
        help = 'PROJECT_ROOT/models/name')
    parser.add_argument("--threads", type = int, 
        default = None, 
        help = 'number of threads to use for the data module')
    parser.add_argument("--debug", type = bool,
        help = 'enable debug')
    return parser.parse_args()

def main(args: argparse.Namespace):    
    # set up the tb logger since that handles creation of the run dir,
    # in case no name was passed
    logger = lightning.loggers.TensorBoardLogger(save_dir = MODEL_ROOT, name = args.name, default_hp_metric = False, version = None)
    # get the actual run log dir
    log_dir = pathlib.Path(logger.log_dir)
    if args.debug:
        import shutil
        shutil.rmtree(log_dir, ignore_errors = True)
    log_dir.mkdir(parents = True, exist_ok = True)

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
    train(config_model = config_model, config_trainer = config_trainer, config_data = config_data,
        log = log_dir, data = args.data.resolve(strict = True), logger = [logger], seed = config_trainer["seed"], verbose = args.debug)

if __name__ == "__main__":
    main(args())
