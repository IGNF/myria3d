from typing import Any, List, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
import os.path as osp
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import IoU
from torchmetrics.functional.classification.iou import _iou_from_confmat

from src.utils import utils

log = utils.get_logger(__name__)


# This is not optimal at the moment, and a single class IoU by phase could be used
# with specific class of interest specified before each logging. But this seems dangerous so
# first tests with num_class objects are performed.


class LogIoUByClass(Callback):
    """
    A Callback to log an IoU for each class.
    We do not log on each step because this could (slightly) mess with IoU computation.
    """

    def __init__(self, classification_dict):
        self.classification_names = classification_dict.values()
        self.num_classes = len(classification_dict)
        self.metric = SingleClassIoU

    def get_all_iou_by_class_object(self):
        """Get a dict with schema {class_name:iou_for_class_name, ...}"""
        iou_dict = {
            name: self.metric(self.num_classes, idx)
            for idx, name in enumerate(self.classification_names)
        }
        return iou_dict

    def on_fit_start(self, trainer, pl_module):
        self.train_iou_by_class_dict = self.get_all_iou_by_class_object()
        self.val_iou_by_class_dict = self.get_all_iou_by_class_object()

    def on_test_start(self, trainer, pl_module):
        self.test_iou_by_class_dict = self.get_all_iou_by_class_object()

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
    ):
        device = outputs["preds"].device
        for class_name, class_iou in self.train_iou_by_class_dict.items():
            # TODO: shoudl we always stay to CPU to preserve GPU ressources ?
            # TODO: move once in on_fit_start using trainer.device / pl_module.device ?
            class_iou = class_iou.to(device)
            class_iou(outputs["preds"], outputs["targets"])
            metric_name = f"train/iou_CLASS_{class_name}"
            self.log(
                metric_name,
                class_iou,
                on_step=False,
                on_epoch=True,
                metric_attribute=metric_name,
            )

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        device = outputs["preds"].device
        for class_name, class_iou in self.val_iou_by_class_dict.items():
            class_iou = class_iou.to(device)
            class_iou(outputs["preds"], outputs["targets"])
            metric_name = f"val/iou_CLASS_{class_name}"
            self.log(
                metric_name,
                class_iou,
                on_step=False,
                on_epoch=True,
                metric_attribute=metric_name,
            )

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        device = outputs["preds"].device
        for class_name, class_iou in self.test_iou_by_class_dict.items():
            class_iou = class_iou.to(device)
            class_iou(outputs["preds"], outputs["targets"])
            metric_name = f"train/iou_CLASS_{class_name}"
            self.log(
                metric_name,
                class_iou,
                on_step=False,
                on_epoch=True,
                metric_attribute=metric_name,
            )


class SingleClassIoU(IoU):
    """
    Custom IoU metrics to log single class IoU using PytorchLighting log system.
    This enables good computation of epoch-level IoU.
    i.e. use the full confusion matrix instead of averaging many step-level IoU.
    Default parameters of IoU are used except for absent_score set to 1.0 and none reduction.
    """

    def __init__(
        self,
        num_classes: int,
        class_of_interest_idx: int,
        ignore_index: Optional[int] = None,
        absent_score: float = 1.0,
        threshold: float = 0.5,
        reduction: str = "none",
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:

        self.class_of_interest_idx = class_of_interest_idx

        super().__init__(
            num_classes,
            ignore_index,
            absent_score,
            threshold,
            reduction,
            compute_on_step,
            dist_sync_on_step,
            process_group,
        )

    # def set_set_class_of_interest(self, class_of_interest_idx):
    #     self.class_of_interest_idx = class_of_interest_idx

    def compute(self):
        """Computes intersection over union (IoU)"""

        iou_no_reduction = _iou_from_confmat(
            self.confmat,
            self.num_classes,
            self.ignore_index,
            self.absent_score,
            self.reduction,
        )
        class_of_interest_iou = iou_no_reduction[self.class_of_interest_idx]
        return class_of_interest_iou
