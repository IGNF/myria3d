import os
import os.path as osp
from typing import Any, List, Optional, Union

import laspy
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn.unpool.knn_interpolate import knn_interpolate

from torch_geometric.nn.pool import knn
from torchmetrics import IoU
from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy

from semantic_val.models.modules.point_net import PointNet as Net
from semantic_val.utils import utils

log = utils.get_logger(__name__)

EPS = 10 ** -5


class WeightedFocalLoss(nn.Module):
    "Weighted version of Focal Loss"

    def __init__(self, weights: torch.Tensor = [0.1, 0.9], gamma: float = 2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = weights
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=1)
        self.eps = EPS

    def forward(self, logits, targets):
        proba = self.softmax(logits)
        n_classes = proba.size(1)
        loss = torch.zeros_like(targets).type(torch.float)
        for i in range(n_classes):
            pi = proba[:, i] * (targets == i) + (1 - proba[:, i]) * (targets != i)
            ai = self.alpha[i]
            loss += -ai * (1 - pi) ** self.gamma * torch.log(pi + self.eps)
        return loss.mean()


# TODO : asbtract PN specific params into a kwargs_model argument.
# TODO: refactor to ClassificationModel if this is not specific to PointNet
class SegmentationModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    :param save_predictions: Set to True to save LAS files with predictions as classification field.
    Only in effect if save_predictions is True.
    """

    def __init__(
        self,
        num_classes: int = 2,
        loss="CrossEntropyLoss",
        lr: float = 0.01,
        save_predictions: bool = False,
        save_train_predictions_every_n_step: int = 50,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.model = Net(hparams=self.hparams)
        self.should_save_preds: bool = save_predictions
        self.save_train_predictions_every_n_step = save_train_predictions_every_n_step
        self.in_memory_tile_id = ""

        self.softmax = nn.Softmax(dim=1)
        percentage_buildings_train_val = 0.0226
        weights = torch.FloatTensor(
            [
                percentage_buildings_train_val,
                1 - percentage_buildings_train_val,
            ]
        )
        if loss == "CrossEntropyLoss":
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        elif loss == "FocalLoss":
            self.criterion = WeightedFocalLoss(weights=weights, gamma=2.0)

        self.train_iou = IoU(num_classes, reduction="none")
        self.val_iou = IoU(num_classes, reduction="none")
        self.test_iou = IoU(num_classes, reduction="none")
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.max_reached_val_iou = -np.inf
        self.val_iou_accumulator: List = []

        self.train_step_global_idx: int = 0
        self.best_reached_train_iou: float = 0.0
        self.train_iou_accumulator: List = []
        self.train_iou_has_improved: bool = False

    def forward(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch)
        logits = knn_interpolate(
            logits,
            batch.pos_copy_subsampled,
            batch.pos_copy,
            batch_x=batch.batch_x,
            batch_y=batch.batch_y,
            k=3,
        )
        return logits

    def step(self, batch: Any):
        targets = batch.y_copy

        logits = self.forward(batch)
        loss = self.criterion(logits, targets)

        proba = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, proba, preds, targets

    def on_fit_start(self):
        log_path = os.getcwd()
        log.info(f"Results and logs saved to {log_path}")
        self.preds_dirpath = osp.join(log_path, "validation_preds")
        os.makedirs(self.preds_dirpath, exist_ok=True)
        self.val_preds_geotiffs_folder = osp.join(self.preds_dirpath, "geotiffs")
        os.makedirs(self.val_preds_geotiffs_folder, exist_ok=True)

        self.experiment = self.logger.experiment[0]
        self.experiment.log_parameter("experiment_logs_dirpath", log_path)

    def on_train_epoch_start(self) -> None:
        self.train_iou_accumulator = []
        self.train_step_global_idx = self.train_step_global_idx + 1
        return super().on_train_start()

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)

        acc = self.train_accuracy(preds, targets)
        iou = self.train_iou(preds, targets)[1]
        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()

        self.train_iou_accumulator.append(iou)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        log.debug(f"Train batch building % = {targets_avg}")
        self.log(
            "train/preds_avg", preds_avg, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/targets_avg",
            targets_avg,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        if (
            self.should_save_preds
            and self.train_step_global_idx % self.save_train_predictions_every_n_step
            == 0
        ):
            self.in_memory_tile_id = None
            self.update_las_with_preds(proba, preds, batch, targets)
            self.save_las_with_preds("train")
            log.info(
                f"Saved train preds to disk for batch number {self.train_step_global_idx}"
            )
            self.in_memory_tile_id = None

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_train_epoch_end(self, unused=None) -> None:
        epoch_train_iou = np.mean(self.train_iou_accumulator)
        if epoch_train_iou > self.best_reached_train_iou:
            self.train_iou_has_improved = True
            self.best_reached_train_iou = epoch_train_iou
        return super().on_train_epoch_end(unused=unused)

    def on_validation_start(self) -> None:
        self.val_iou_accumulator = []
        if not self.train_iou_has_improved:
            log.info("Skipping validation until train IoU increases.\n")
        return super().on_train_start()

    def validation_step(self, batch: Any, batch_idx: int):
        if not self.train_iou_has_improved:
            self.log("val/iou", -1.0, on_step=True, on_epoch=True, prog_bar=True)
            return None

        loss, _, proba, preds, targets = self.step(batch)
        acc = self.val_accuracy(preds, targets)
        iou = self.val_iou(preds, targets)[1]
        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()

        self.val_iou_accumulator.append(iou)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val/preds_avg", preds_avg, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "val/targets_avg",
            targets_avg,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        if self.should_save_preds:
            self.update_las_with_preds(proba, preds, batch, targets)

        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def on_validation_end(self):
        """Save the last unsaved predicted las and keep track of best IoU"""
        if self.train_iou_has_improved:
            log.info(
                f"Saved validation preds to disk after train step {self.train_step_global_idx}"
            )
            self.save_las_with_preds("val")
            val_iou = np.mean(self.val_iou_accumulator)
            self.max_reached_val_iou = max(val_iou, self.max_reached_val_iou)
            self.experiment.log_metric("val/max_iou", self.max_reached_val_iou)

        self.train_iou_has_improved = False

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        acc = self.test_accuracy(preds, targets)
        iou = self.test_iou(preds, targets)[1]
        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()

        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/iou", iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "test/preds_avg", preds_avg, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "test/targets_avg",
            targets_avg,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        if self.should_save_preds:
            log.info(
                f"Saving test preds to disk after train step {self.train_step_global_idx}"
            )
            self.update_las_with_preds(proba, preds, batch, targets)

        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def on_test_end(self):
        """Save the last unsaved predicted las and keep track of best IoU"""
        self.save_las_with_preds("test")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
        )

    def update_las_with_preds(
        self,
        proba: torch.Tensor,
        preds: torch.Tensor,
        batch: Batch,
        targets: torch.Tensor,
        phase: str = "val",  # val, train or test
    ):
        """Save the predicted classes in las format with position."""
        batch_size = len(np.unique(batch.batch))
        for sample_idx in range(batch_size):
            elem_tile_id = batch.tile_id[sample_idx]
            if self.in_memory_tile_id != elem_tile_id:
                if self.in_memory_tile_id:
                    self.save_las_with_preds(phase)
                self.in_memory_tile_id = elem_tile_id
                self.las_with_preds = laspy.read(batch.filepath[sample_idx])
                param = laspy.ExtraBytesParams(name="building_proba", type=float)
                self.las_with_preds.add_extra_dim(param)
                param = laspy.ExtraBytesParams(
                    name="classification_confusion", type=int
                )
                self.las_with_preds.add_extra_dim(param)

                # TODO: consider setting this to np.nan or equivalent to capture incomplete predictions.
                self.las_with_preds.classification[:] = 0
                self.val_las_pos = np.asarray(
                    [
                        self.las_with_preds.x,
                        self.las_with_preds.y,
                        self.las_with_preds.z,
                    ],
                    dtype=np.float32,
                ).transpose()
                self.val_las_pos = torch.from_numpy(self.val_las_pos)

            elem_pos = batch.pos_copy[batch.batch_y == sample_idx]
            elem_preds = preds[batch.batch_y == sample_idx]
            elem_proba = proba[batch.batch_y == sample_idx][:, 1]
            elem_targets = targets[batch.batch_y == sample_idx]

            assign_idx = knn(self.val_las_pos, elem_pos, k=1, num_workers=1)[1]

            self.las_with_preds.classification[assign_idx] = elem_preds
            elem_predsdiff = elem_preds + 2 * elem_targets
            self.las_with_preds.classification_confusion[assign_idx] = elem_predsdiff
            self.las_with_preds.building_proba[assign_idx] = elem_proba.detach()
        return

    def save_las_with_preds(self, phase):
        """After inference of classification in self.las_with_predictions, save:
        - The LAS with updated classification
        - A GeoTIFF of the classification, which is logged into Comet as well.
        """
        output_path = osp.join(
            self.preds_dirpath,
            f"{phase}_{self.in_memory_tile_id}.las",
        )
        self.las_with_preds.write(output_path)
        log.info(f"Predictions save path : {output_path}.")
