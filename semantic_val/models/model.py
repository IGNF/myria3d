import os
import os.path as osp
from typing import Any, List, Optional, Union

import laspy
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
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
    """
    Weighted version of Focal Loss.
    We normalize the loss by the nb of samples with rare class, inspired by original Focal Loss paper.
    """

    def __init__(self, weights: torch.Tensor = [0.1, 0.9], gamma: float = 2.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = weights
        self.gamma = gamma
        self.softmax = nn.Softmax(dim=1)
        self.eps = EPS
        self.n_classes = len(weights)
        self.rare_class_dim = 1

    def forward(self, logits, targets):
        assert logits.size(1) == self.n_classes
        proba = self.softmax(logits)
        loss = torch.zeros_like(targets).type(torch.float)
        for i in range(self.n_classes):
            ti = targets == i
            pi = proba[:, i] * ti
            ai = self.alpha[i]
            loss += -(ti * ai) * (1 - pi) ** self.gamma * torch.log(pi + self.eps)
        loss = loss.sum() / (targets == self.rare_class_dim).sum()
        return loss


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
        n_classes: int = 2,
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

        self.train_iou = IoU(n_classes, reduction="none")
        self.val_iou = IoU(n_classes, reduction="none")
        self.test_iou = IoU(n_classes, reduction="none")
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        # TODO: Abstract the tracking of max reached in a separate hook
        self.max_reached_val_iou = -np.inf
        self.val_iou_accumulator: List = []

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

    def on_train_epoch_start(self) -> None:
        self.train_iou_accumulator = []
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

        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

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

        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

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
