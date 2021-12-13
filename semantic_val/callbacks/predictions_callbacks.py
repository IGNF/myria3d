from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
import os

import laspy
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import os.path as osp
import torch
from torch_geometric.nn.pool import knn
from semantic_val.datamodules.processing import DataHandler

from semantic_val.utils import utils

log = utils.get_logger(__name__)


class SavePreds(Callback):
    """
    A Callback to save predictions back to original LAS file.
    Keep a full LAS tile in memory until it changes and thus must be saved.
    Added channels: BuildingProba, BuildingPreds
    """

    def __init__(
        self,
        save_predictions: bool = False,
        save_train_predictions_every_n_step: int = 10 ** 6,
    ):
        self.in_memory_tile_filepath = ""
        self.save_predictions = save_predictions
        self.save_train_predictions_every_n_step = save_train_predictions_every_n_step

        self.data_handler = DataHandler()

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """Setup logging functionnalities ; create the outputs dir."""

        self.experiment = trainer.logger.experiment[0]
        log_path = os.getcwd()
        log.info(f"Saving results and logs to {log_path}")
        self.experiment.log_parameter("experiment_logs_dirpath", log_path)

        if self.save_predictions:
            self.data_handler.preds_dirpath = osp.join(log_path, "validation_preds")
            os.makedirs(self.data_handler.preds_dirpath, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_predictions:
            reached_train_saving_step = (
                trainer.global_step % self.save_train_predictions_every_n_step == 0
            )
            if reached_train_saving_step:
                self.data_handler.update_las_with_proba(outputs, "train")
                log.debug(
                    f"Saving train preds to disk for batch number {trainer.global_step}"
                )
                self.data_handler.save_las_with_proba_and_close("train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_predictions:
            if outputs is not None:
                self.data_handler.update_las_with_proba(outputs, "val")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.save_predictions:
            log.debug(
                f"Saving validation preds to disk after train step {trainer.global_step}.\n"
            )
            self.data_handler.save_las_with_proba_and_close("val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_predictions:
            log.debug(
                f"Saving test preds to disk after train step {trainer.global_step}"
            )
            self.data_handler.update_las_with_proba(outputs, "test")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.save_predictions:
            self.data_handler.save_las_with_proba_and_close("test")

    def on_predict_batch_end(
        self,  # predict#
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        self.data_handler.update_las_with_proba(outputs, "predict")
