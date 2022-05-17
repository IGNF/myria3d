try:
    # It is safer to import comet before all other imports.
    import comet_ml  # noqa
except ImportError:
    print(
        "Warning: package comet_ml not found. This may break things if you use a comet callback."
    )

import copy
import os
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from myria3d.models.model import Model

from myria3d.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Trainer:
    """Training pipeline (+ Test, + Finetuning)

    Instantiates all PyTorch Lightning objects from config, then perform one of the following
    task based on parameter `task.task_name`:

    fit:
        Fits a neural network - train on a prepared training set and validate on a prepared validation set.
        Optionnaly, resume a checkpointed training by specifying config.model.ckpt_path.

    test:
        Tests a trained neural network on the test dataset of a prepared dataset (i.e. the `test` subdir
        which contains LAS files with a classification).

    finetune:
        Finetunes a checkpointed neural network on a prepared dataset, which muste be specified
        in config.model.ckpt_path.
        In contrast to using fit, finetuning resumes training with altered conditions. This leads to a new,
        distinct training, and training state is reset (e.g. epoch starts from 0).

    Typical use case are

        - a different learning rate (config.model.lr) or a different scheduler (e.g. stronger config.model.lr_scheduler.patience)
        - a different number of classes to predict, in order to e.g. specialize a base model. \
        This is done by specifying a new config.datamodule.dataset_description as well as the corresponding config.model.num_classes. \
        for RecudeLROnPlateau scheduler). Additionnaly, a specific callback must be activated to change neural net output layer \
        after loading its weights. See configs/experiment/RandLaNetDebugFineTune.yaml for an example.


    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Trainer: lightning trainer.

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

    if "fit" in task_name:
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
            log.info(f"Best lr with auto_lr_find is {new_lr}")

        log.info("Starting training and validating!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=config.model.ckpt_path
        )
        log.info(f"Best checkpoint:\n{trainer.checkpoint_callback.best_model_path}")
        log.info("End of training and validating!")

    if "test" in task_name:
        log.info("Starting testing!")
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path=config.model.ckpt_path
        )
        log.info("End of testing!")

    if "finetune" in task_name:
        log.info("Starting finetuning pretrained model on new data!")
        # Instantiates the Model but overwrites everything with current config,
        # except module related params (nnet architecture)
        kwargs_to_override = copy.deepcopy(model.hparams)
        NEURAL_NET_ARCHITECTURE_CONFIG_GROUP = "neural_net"
        kwargs_to_override = {
            key: value
            for key, value in kwargs_to_override.items()
            if NEURAL_NET_ARCHITECTURE_CONFIG_GROUP not in key
        }
        model = Model.load_from_checkpoint(config.model.ckpt_path, **kwargs_to_override)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
        log.info(f"Best checkpoint:\n{trainer.checkpoint_callback.best_model_path}")
        log.info("End of training and validating!")

    # Returns the trainer for access to everything that was calculated.
    return trainer
