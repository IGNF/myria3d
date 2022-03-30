from typing import Any, Optional
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torchmetrics import MaxMetric
from lidar_multiclass.models.modules.randla_net import RandLANet
from lidar_multiclass.models.modules.point_net import PointNet
from lidar_multiclass.utils import utils

log = utils.get_logger(__name__)


class Model(LightningModule):
    """This LightningModule implements the logic for model trainin, validation, tests, and prediction.

    It is fully initialized by named parameters for maximal flexibility with hydra configs.

    During training and validation, IoU is calculed based on sumbsampled points only, and is therefore
    an approximation.
    At test time, IoU is calculated considering all the points. To keep this module light, a callback
    takes care of the interpolation of predictions between all points.


    Read the Pytorch Lightning docs:
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

        neural_net_class = self.get_neural_net_class(self.hparams.neural_net_class_name)
        self.model = neural_net_class(self.hparams.neural_net_hparams)

        self.softmax = nn.Softmax(dim=1)

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_iou = self.hparams.iou()
            self.val_iou = self.hparams.iou()
            self.val_iou_best = MaxMetric()
        if stage == "test":
            self.test_iou = self.hparams.iou()
        if stage != "predict":
            self.criterion = self.hparams.criterion

    def forward(self, batch: Batch) -> torch.Tensor:
        logits = self.model(batch)
        return logits

    def step(self, batch: Any):
        logits = self.forward(batch)
        targets = batch.y
        loss = self.criterion(logits, targets)
        return loss, logits, targets

    def on_fit_start(self) -> None:
        self.experiment = self.logger.experiment[0]

    def training_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
        self.train_iou(preds, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train/iou", self.train_iou, on_step=True, on_epoch=True, prog_bar=True
        )
        return {
            "loss": loss,
            "logits": logits,
            "targets": targets,
        }

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logits, targets = self.step(batch)
        preds = torch.argmax(logits, dim=1)
        self.val_iou(preds, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        self.log("val/iou", self.val_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {
            "loss": loss,
            "logits": logits,
            "targets": targets,
        }

    def validation_epoch_end(self, outputs):
        iou = self.val_iou.compute()
        self.val_iou_best.update(iou)
        self.log(
            "val/iou_best", self.val_iou_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        logits = self.forward(batch)
        return {"logits": logits, "batch": batch}

    def predict_step(self, batch: Any):
        logits = self.forward(batch)
        return {"batch": batch, "logits": logits}

    def get_neural_net_class(self, class_name):
        """A Class Factory to class of neural net based on class name.

        :meta private:

        """
        for neural_net_class in [PointNet, RandLANet]:
            if class_name in neural_net_class.__name__:
                return neural_net_class
        raise KeyError(f"Unknown class name {class_name}")

    def configure_optimizers(self):
        self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        try:
            lr_scheduler = self.hparams.lr_scheduler(optimizer)
        except:
            # OneCycleLR needs optimizer and max_lr
            lr_scheduler = self.hparams.lr_scheduler(optimizer, self.lr)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": self.hparams.monitor,
        }

        return config
