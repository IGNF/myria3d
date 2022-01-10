from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Batch

from torchmetrics import IoU

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
        num_classes: int = 2,  # TODO: update for multiclass
        alpha: float = 0.25,
        lr: float = 0.001,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.lr = lr
        self.save_hyperparameters()

        # TODO: use hydra instantiate here instead
        self.model_architecture = kwargs.get("model_architecture", "randla_net")
        model_class = MODEL_ZOO[self.model_architecture]
        self.model = model_class(hparams=self.hparams)
        self.softmax = nn.Softmax(dim=1)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_iou = IoU(num_classes=num_classes, threshold=0.5, absent_score=1.0)
        self.val_iou = IoU(num_classes=num_classes, threshold=0.5, absent_score=1.0)
        self.test_iou = IoU(num_classes=num_classes, threshold=0.5, absent_score=1.0)

        # TODO: remove this after tests on GPU
        # self.metrics = [self.train_iou, self.val_iou, self.test_iou]

    def forward(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch)
        return logits

    def step(self, batch: Any):
        logits = self.forward(batch)
        targets = batch.y
        loss = self.criterion(logits, targets)
        with torch.no_grad():
            proba = self.softmax(logits)
            preds = torch.argmax(logits, dim=1)
        return loss, logits, proba, preds, targets

    def on_fit_start(self) -> None:
        self.experiment = self.logger.experiment[0]
        # TODO: remove this after tests on GPU
        # for metric in self.metrics:
        #     metric = metric.to(self.device)
        # assert all(metric.device == self.device for metric in self.metrics)

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.train_iou(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True
        )
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.val_iou(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/iou", self.val_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True)
        self.test_iou(preds, targets)
        self.log("test/iou", self.test_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def predict_step(self, batch: Any):
        logits = self.forward(batch)
        proba = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)
        return {"batch": batch, "proba": proba, "preds": preds}

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
