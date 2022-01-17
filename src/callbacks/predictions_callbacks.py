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
from src.datamodules.processing import DataHandler

from src.utils import utils

log = utils.get_logger(__name__)


class SavePreds(Callback):
    """
    A Callback to save predictions back to original LAS file, into a new BuildingProba channel.
    Keep a full LAS tile in memory until it changes and thus must be saved.
    """

    def __init__(
        self,
        save_predictions: bool = False,
    ):
        self.save_predictions = save_predictions

        if self.save_predictions:
            log_path = os.getcwd()
            preds_dirpath = osp.join(log_path, "predictions")
            self.data_handler = DataHandler(output_dir=preds_dirpath)

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """Setup logging functionnalities ; create the outputs dir."""
        # TODO: log the dirpath elsewhere in code (train.py before fit is called?),
        self.experiment = trainer.logger.experiment[0]
        log_path = os.getcwd()
        log.info(f"Saving results and logs to {log_path}")
        self.experiment.log_parameter("experiment_logs_dirpath", log_path)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_predictions and outputs is not None:
            self.data_handler.update_with_inference_outputs(outputs, "val")

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
            self.data_handler.update_with_inference_outputs(outputs, "test")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.save_predictions:
            self.data_handler._interpolate_classification_and_save("val")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.save_predictions:
            self.data_handler._interpolate_classification_and_save("test")
