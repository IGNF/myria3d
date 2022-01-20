import os
import comet_ml
from typing import List, Optional

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger
    )

    # Send some parameters from config to all lightning loggers
    log.info("Logging hyperparameters!")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    task_name = config.task.get("task_name")

    if config.trainer.auto_lr_find:
        log.info("Finding best lr with auto_lr_find!")
        # Run learn ing rate finder
        lr_finder = trainer.tuner.lr_find(
            model,
            datamodule=datamodule,
            min_lr=1e-6,
            max_lr=3,
            num_training=200,
            mode="exponential",
        )

        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        os.makedirs("./hpo/", exist_ok=True)
        fig.savefig(
            f"./hpo/lr_range_test_best_{new_lr:.5}.png",
        )

        # update hparams of the model
        model.hparams.lr = new_lr

    if "fit" in task_name:
        log.info("Starting training and validating!")
        trainer.fit(model=model, datamodule=datamodule)
        log.info(f"Best checkpoint:\n{trainer.checkpoint_callback.best_model_path}")
        log.info("End of training and validating!")

    if "test" in task_name:
        log.info("Starting testing!")
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path=trainer.resume_from_checkpoint
        )  # ckpt_path
        log.info("End of testing!")
