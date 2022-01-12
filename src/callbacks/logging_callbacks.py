from typing import Any, List, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
import os.path as osp
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

from src.utils import utils

log = utils.get_logger(__name__)


class LogIoUByClass(Callback):
    """
    A Callback to log an IoU for each class.
    We do not log on each step because this could (slightly) mess with IoU computation.
    """

    def __init__(self, iou_by_class, classification_dict):
        self.classification_names = classification_dict.values()

        self.train_iou_by_class = iou_by_class()
        self.val_iou_by_class = iou_by_class()
        self.test_iou_by_class = iou_by_class()

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """Setup logging functionnalities."""
        self.experiment = trainer.logger.experiment[0]

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.train_iou_by_class = self.train_iou_by_class.to(outputs["preds"].device)

        iou_values = self.train_iou_by_class(outputs["preds"], outputs["targets"])
        for name, value in zip(self.classification_names, iou_values):
            self.log(f"train/iou_CLASS_{name}", value, on_step=False, on_epoch=True)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.val_iou_by_class = self.val_iou_by_class.to(outputs["preds"].device)

        iou_values = self.val_iou_by_class(outputs["preds"], outputs["targets"])
        for name, value in zip(self.classification_names, iou_values):
            self.log(f"val/iou_CLASS_{name}", value, on_step=False, on_epoch=True)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.test_iou_by_class = self.test_iou_by_class.to(outputs["preds"].device)

        iou_values = self.test_iou_by_class(outputs["preds"], outputs["targets"])
        for name, value in zip(self.classification_names, iou_values):
            self.log(f"test/iou_CLASS_{name}", value, on_step=False, on_epoch=True)

    # def on_validation_end(
    #     self, trainer: pl.Trainer, pl_module: pl.LightningModule
    # ) -> None:

    # def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
    #     if self.save_predictions:
    #         self.data_handler.interpolate_classification_and_save("test")
