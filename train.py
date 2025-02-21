import os
import hydra
import wandb
import pytorch_lightning as pl
from noksr.callback import *
from importlib import import_module
from noksr.data.data_module import DataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy



def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, gpu_cache_clean_monitor, lr_monitor]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)

    output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(output_path, exist_ok=True)

    print("==> initializing data ...")
    data_module = DataModule(cfg)

    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=output_path)

    print("==> initializing monitor ...")
    callbacks = init_callbacks(cfg)


    print("==> initializing trainer ...")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.model.trainer, strategy=DDPStrategy(find_unused_parameters=False))

    print("==> initializing model ...")
    model = getattr(import_module("noksr.model"), cfg.model.network.module)(cfg)


    print("==> start training ...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)

if __name__ == '__main__':
    main()