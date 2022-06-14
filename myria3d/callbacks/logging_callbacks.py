from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torchmetrics import JaccardIndex
from torchmetrics.functional.classification.jaccard import _jaccard_from_confmat
from myria3d.models.interpolation import Interpolator
from myria3d.utils import utils

log = utils.get_logger(__name__)


# Training was not lenghtend so we keep "as-is" for now, but this
# is not optimal at the moment, and a single class JaccardIndex by phase could
# be used # with specific class of interest specified before each logging.


class LogIoUByClass(Callback):
    """
    A Callback to log JaccardIndex for each class.
    """

    def __init__(self, classification_dict: Dict[int, str], interpolator: Interpolator):
        self.classification_names = classification_dict.values()
        self.num_classes = len(classification_dict)
        self.metric = SingleClassIoU
        self.itp = interpolator

    def get_all_iou_by_class_object(self):
        """Get a dict with schema {class_name:iou_for_class_name, ...}"""
        iou_dict = {
            name: self.metric(self.num_classes, idx)
            for idx, name in enumerate(self.classification_names)
        }
        return iou_dict

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Setup IoU torchmetrics objects for train and val phases."""
        self.train_iou_by_class_dict = self.get_all_iou_by_class_object()
        self.val_iou_by_class_dict = self.get_all_iou_by_class_object()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Setup IoU torchmetrics objects for test phase."""
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
        """Log IoU for each class."""
        logits = outputs["logits"]
        targets = batch.y
        self.log_iou(logits, targets, "train", self.train_iou_by_class_dict)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Log IoU for each class."""
        logits = outputs["logits"]
        targets = batch.y
        self.log_iou(logits, targets, "val", self.val_iou_by_class_dict)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        """Log IoU for each class."""
        logits = outputs["logits"]
        targets = outputs["targets"]
        self.log_iou(logits, targets, "test", self.test_iou_by_class_dict)

    def log_iou(self, logits, targets, phase: str, iou_dict):
        device = logits.device
        preds = torch.argmax(logits, dim=1)
        for class_name, class_iou in iou_dict.items():
            class_iou = class_iou.to(device)
            class_iou(preds, targets)
            metric_name = f"{phase}/iou_CLASS_{class_name}"
            self.log(
                metric_name,
                class_iou,
                on_step=False,
                on_epoch=True,
                metric_attribute=metric_name,
            )


class SingleClassIoU(JaccardIndex):
    """
    Custom JaccardIndex metrics to log single class JaccardIndex using PytorchLighting log system.
    This enables good computation of epoch-level JaccardIndex.
    i.e. use the full confusion matrix instead of averaging many step-level JaccardIndex.
    Default parameters of JaccardIndex are used except for absent_score set to 1.0 and none reduction.

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

    def compute(self):
        """Computes intersection over union (JaccardIndex)"""

        iou_no_reduction = _jaccard_from_confmat(
            self.confmat,
            self.num_classes,
            self.ignore_index,
            self.absent_score,
            self.reduction,
        )
        class_of_interest_iou = iou_no_reduction[self.class_of_interest_idx]
        return class_of_interest_iou
