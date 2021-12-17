from typing import Any, Optional

import torch
from torch import Tensor
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch
from torch_geometric.nn.unpool.knn_interpolate import knn_interpolate

from torchmetrics import IoU
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional.classification.iou import _iou_from_confmat

from semantic_val.models.modules.point_net import PointNet
from semantic_val.models.modules.randla_net import RandLANet
from semantic_val.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = {"point_net": PointNet, "randla_net": RandLANet}

EPS = 10 ** -5


class Model(LightningModule):
    """
    A LightningModule organizesm your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model_architecture: str = "point_net",
        n_classes: int = 2,
        loss: str = "CrossEntropyLoss",
        alpha: float = 0.25,
        lr: float = 0.001,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.lr = lr
        self.save_hyperparameters()

        model_class = MODEL_ZOO[model_architecture]
        self.model = model_class(hparams=self.hparams)
        self.softmax = nn.Softmax(dim=1)

        weights = torch.FloatTensor([alpha, 1 - alpha])
        if loss == "CrossEntropyLoss":
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights)
        elif loss == "FocalLoss":
            self.criterion = WeightedFocalLoss(weights=weights, gamma=2.0)

        self.train_iou = BuildingsIoU()
        self.val_iou = BuildingsIoU()
        self.test_iou = BuildingsIoU()
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.metrics = [
            self.train_iou,
            self.val_iou,
            self.test_iou,
            self.train_accuracy,
            self.val_accuracy,
            self.test_accuracy,
        ]

    # TODO: to avoid costly KNN, return logits directly.
    # TODO: use a separate logic for prediction
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

    # TODO: to avoid costly KNN, compare with batch.y directly (subsampled already)
    # Deal with interpolation in DataHandler : use a K=3, be sure that every point gets a proba.
    def step(self, batch: Any):
        targets = batch.y_copy

        logits = self.forward(batch)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        with torch.no_grad():
            proba = self.softmax(logits)
        return loss, logits, proba, preds, targets

    def on_fit_start(self) -> None:
        self.experiment = self.logger.experiment[0]
        for metric in self.metrics:
            metric = metric.to(self.device)
        assert all(metric.device == self.device for metric in self.metrics)

    # TODO: decide if returning preds is needed
    def training_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        self.train_accuracy(preds, targets)
        self.log(
            "train/acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )

        self.train_iou(preds, targets)
        self.log(
            "train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True
        )

        targets_avg = (targets * 1.0).mean().item()
        preds_avg = (preds * 1.0).mean().item()
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
            "targets": targets,
            "batch": batch,
        }

    # TODO: decide if returning preds is needed
    def validation_step(self, batch: Any, batch_idx: int, dataset_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        self.val_accuracy(preds, targets)
        self.log(
            "val/acc", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )

        self.val_iou(preds, targets)
        self.log("val/iou", self.val_iou, on_step=True, on_epoch=True, prog_bar=True)

        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()

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
            "targets": targets,
            "batch": batch,
        }

    def test_step(self, batch: Any, batch_idx: int, dataset_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        self.test_accuracy(preds, targets)
        self.log(
            "test/acc", self.test_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )

        self.test_iou(preds, targets)
        self.log("test/iou", self.test_iou, on_step=True, on_epoch=True, prog_bar=True)

        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()
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
            "targets": targets,
            "batch": batch,
        }

    @torch.no_grad()
    def predict_step(self, batch: Any):
        # TODO: may ned to use other than forward.
        logits = self.forward(batch)
        proba = self.softmax(logits)
        return {
            "batch": batch,
            "proba": proba,
        }

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        if self.hparams.reduce_lr_on_plateau.activate:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.reduce_lr_on_plateau.factor,
                patience=self.hparams.reduce_lr_on_plateau.patience,  # scheduler called on training epoch !
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=self.hparams.reduce_lr_on_plateau.cooldown,
                min_lr=0,
                eps=1e-08,
                verbose=False,
            )
            log.info("ReduceLROnPlateau: activated")
            config = {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss_epoch",
            }
            return config
        return optimizer


class WeightedFocalLoss(nn.Module):
    """
    Weighted version of Focal Loss.
    We normalize in part the loss by the nb of samples with rare class, inspired by original Focal Loss paper.
    This is important so that the loss is properly scaled.
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
        n_points = logits.size(0)
        n_points_rare_class = (targets == self.rare_class_dim).sum()
        normalization_factor = (n_points_rare_class + n_points / 100) / 2
        loss = loss.sum() / normalization_factor
        return loss


class BuildingsIoU(IoU):
    """Custom IoU metrics to log building IoU only using PytorchLighting log system."""

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            num_classes=2,
            threshold=0.5,
            reduction="none",
            absent_score=1.0,
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

    def compute(self) -> Tensor:
        """Computes intersection over union (IoU)"""
        iou_no_reduction = _iou_from_confmat(
            self.confmat,
            self.num_classes,
            self.ignore_index,
            self.absent_score,
            self.reduction,
        )
        iou_building = iou_no_reduction[1]
        return iou_building
