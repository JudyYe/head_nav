import os.path as osp

import pytorch_lightning as pl
import torch
import yaml
from hydra import main
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import DataModule
from dataset_config import dataset_config
from nnutils.logger import LoggerCallback, build_logger
import importlib
from model.network import CameraCVAE
from nnutils import model_utils


def instantiate_module(module_cfg, **kwargs):
    target = module_cfg.target
    module_name = ".".join(target.split(".")[:-1])
    class_name = target.split(".")[-1]
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    param = module_cfg.param
    return class_(**param, **kwargs)


@main(config_path="config", config_name="train", version_base=None)
def train(cfg):
    pl.seed_everything(123)

    # Load dataset config
    dataset_cfg = dataset_config()

    # Setup training and validation datasets
    datamodule = DataModule(cfg, dataset_cfg)
    datamodule.setup()
    len_train = len(datamodule.train_dataloader())

    with open(osp.join(cfg.exp_dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_yaml(cfg), f)
    with open(osp.join(cfg.exp_dir, "dataset_config.yaml"), "w") as f:
        f.write(dataset_cfg.dump())

    model = CameraCVAE(cfg)
    if cfg.checkpoint is not None:
        miss_keys, _, _ = model_utils.load_my_state_dict(
            model, torch.load(cfg.checkpoint)["state_dict"]
        )
        print("Missed keys:", miss_keys)

    logger = build_logger(cfg.expname, cfg.exp_dir, log=cfg.log)

    train_kwargs = {}
    val_kwargs = {}
    if cfg.save_every < len_train:
        train_kwargs["every_n_train_steps"] = cfg.save_every
        val_kwargs["val_check_interval"] = cfg.save_every
    else:
        train_kwargs["every_n_epochs"] = cfg.save_every // len_train
        val_kwargs["check_val_every_n_epoch"] = cfg.save_every // len_train

    if cfg.iters is not None:
        val_kwargs["max_steps"] = cfg.iters
    else:
        val_kwargs["max_epochs"] = cfg.epochs
    checkpoint_callback = ModelCheckpoint(
        dirpath=osp.join(cfg.exp_dir, "checkpoints"),
        monitor="step",
        mode="max",
        save_top_k=3,
        save_last=True,
        **train_kwargs,
    )

    callbacks = [checkpoint_callback, LoggerCallback()]

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=100,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        limit_val_batches=1,
        gradient_clip_val=cfg.grad_clip,
        **val_kwargs,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path="last",
    )


if __name__ == "__main__":
    train()
