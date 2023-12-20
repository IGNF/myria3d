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
from pytorch_lightning.loggers.logger import Logger

from myria3d.models.model import Model
from myria3d.utils import utils
from run import TASK_NAMES

log = utils.get_logger(__name__)

NEURAL_NET_ARCHITECTURE_CONFIG_GROUP = "neural_net"


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
        This is done by specifying a new config.dataset_description as well as the corresponding config.model.num_classes. \
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
        for cb_conf in config.callbacks.values():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[Logger] = []
    if "logger" in config:
        for lg_conf in config.logger.values():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

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
    if task_name == TASK_NAMES.FIT.value:
        if config.task.auto_lr_find:
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
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.model.ckpt_path)
        log.info(f"Best checkpoint:\n{trainer.checkpoint_callback.best_model_path}")
        log.info("End of training and validating!")
    if task_name in [TASK_NAMES.FIT.value, TASK_NAMES.TEST.value]:
        log.info("Starting testing!")
        if trainer.checkpoint_callback.best_model_path:
            log.info(
                f"Test will use just-trained best model checkpointed at \n {trainer.checkpoint_callback.best_model_path}"
            )
            config.model.ckpt_path = trainer.checkpoint_callback.best_model_path
        log.info(f"Test will use specified model checkpointed at \n {config.model.ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=config.model.ckpt_path)
        log.info("End of testing!")

    if task_name == TASK_NAMES.FINETUNE.value:
        log.info("Starting finetuning pretrained model on new data!")
        # Instantiates the Model but overwrites everything with current config,
        # except module related params (nnet architecture)
        kwargs_to_override = copy.deepcopy(model.hparams)
        kwargs_to_override.pop(
            NEURAL_NET_ARCHITECTURE_CONFIG_GROUP, None
        )  # removes that key if it's there
        model = Model.load_from_checkpoint(config.model.ckpt_path, **kwargs_to_override)
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
        log.info(f"Best checkpoint:\n{trainer.checkpoint_callback.best_model_path}")
        log.info("End of training and validating!")

    # Returns the trainer for access to everything that was calculated.
    return trainer
