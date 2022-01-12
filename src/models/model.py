from typing import Any
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from src.models.modules.randla_net import RandLANet
from src.models.modules.point_net import PointNet
from src.utils import utils

log = utils.get_logger(__name__)


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
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        neural_net_class = self.get_neural_net_class_name(
            self.hparams.neural_net_class_name
        )
        self.model = neural_net_class(self.hparams.neural_net_hparams)

        self.lr = self.hparams.lr  # aliasing for Pytorch Lightning
        self.criterion = self.hparams.criterion

        self.train_iou = self.hparams.iou()
        self.val_iou = self.hparams.iou()
        self.test_iou = self.hparams.iou()

        self.softmax = nn.Softmax(dim=1)
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

    def get_neural_net_class_name(self, class_name):
        """Access class of neural net based on class name."""
        for neural_net_architecture in [PointNet, RandLANet]:
            if class_name in neural_net_architecture.__name__:
                return neural_net_architecture
        raise KeyError(f"Unknown class name {class_name}")

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters(), lr=self.lr)
        lr_scheduler = self.hparams.lr_scheduler(optimizer)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": self.hparams.monitor,
        }
        log.info(f"Scheduler config:\n{config}")

        return config
