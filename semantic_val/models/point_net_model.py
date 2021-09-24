from typing import Any, List, Union

import torch
from pytorch_lightning import LightningModule
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset
from torchmetrics import IoU
from torchmetrics.classification.accuracy import (  # change for https://torchmetrics.readthedocs.io/en/latest/references/modules.html?highlight=iou#iou later
    Accuracy,
)

from semantic_val.models.modules.point_net import PointNet as Net


class PointNetModel(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
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
        num_classes: int = 2,
        MLP1_channels: List[int] = [3, 32, 32],
        MLP2_channels: List[int] = [32, 64, 64],
        MLP3_channels: List[int] = [64 + 32, 128, 128, 64, 16, 2],
        subsampling_size: int = 1000,
        lr: float = 0.001,
        # weight_decay: float = 0.0005,  # TODO: UNCOMMENT AT SOME POINT TO REGULARIZE
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = Net(hparams=self.hparams)

        # loss function
        # TODO: parametrize : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_iou = IoU(num_classes, reduction="none")
        self.val_iou = IoU(num_classes, reduction="none")
        self.test_iou = IoU(num_classes, reduction="none")

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, data: torch.Tensor):
        return self.model(data)

    def step(self, batch: Any):
        y = batch.y
        logits = self.forward(batch)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_accuracy(preds, targets)
        iou = self.train_iou(preds, targets)[1]
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        iou = self.val_iou(preds, targets)[1]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
        }

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        iou = self.test_iou(preds, targets)[1]

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

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
