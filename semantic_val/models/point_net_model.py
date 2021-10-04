import os
import os.path as osp
from typing import Any, List, Union

import laspy
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.nn.pool import knn
from torchmetrics import IoU
from torchmetrics.classification.accuracy import Accuracy

from semantic_val.models.modules.point_net import PointNet as Net
from semantic_val.utils import utils

log = utils.get_logger(__name__)


# TODO : asbtract PN specific params into a kwargs_model argument.
# TODO: refactor to ClassificationModel if this is not specific to PointNet
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
        save_predictions: bool = False,
        in_memory_tile_id: str = "",
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.model = Net(hparams=self.hparams)

        self.save_predictions = save_predictions
        self.in_memory_tile_id = ""

        # TODO: parametrize : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=crossentropyloss#
        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 1.0]))
        self.train_iou = IoU(num_classes, reduction="none")
        self.val_iou = IoU(num_classes, reduction="none")
        self.test_iou = IoU(num_classes, reduction="none")
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def forward(self, data: torch.Tensor):
        return self.model(data)

    def step(self, batch: Any):
        targets = batch.y

        logits = self.forward(batch)
        loss = self.criterion(logits, targets)

        proba = self.softmax(logits)
        preds = torch.argmax(logits, dim=1)
        return loss, logits, proba, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, preds, targets = self.step(batch)
        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()
        log.debug(f"Train batch building % = {targets_avg}")
        acc = self.train_accuracy(preds, targets)
        iou = self.train_iou(preds, targets)[1]
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/preds_avg", preds_avg, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/targets_avg",
            targets_avg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)

        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()
        acc = self.val_accuracy(preds, targets)
        iou = self.val_iou(preds, targets)[1]

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/preds_avg", preds_avg, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val/targets_avg",
            targets_avg,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {
            "loss": loss,
            "proba": proba,
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def on_validation_start(self):
        log_path = os.getcwd()
        self.val_preds_folder = osp.join(log_path, "validation_preds")
        os.makedirs(self.val_preds_folder, exist_ok=True)

    def validation_step_end(self, output: dict):
        """Save the predicted classes in las format with position."""
        # see https://laspy.readthedocs.io/en/latest/complete_tutorial.html
        if self.save_predictions:
            proba = output["proba"]
            preds = output["preds"]
            batch = output["batch"]
            for sample_idx in range(len(np.unique(batch.batch))):
                elem_tile_id = batch.tile_id[sample_idx]
                if self.in_memory_tile_id != elem_tile_id:
                    if self.in_memory_tile_id:
                        output_path = osp.join(
                            self.val_preds_folder,
                            f"{self.in_memory_tile_id}.las",
                        )
                        self.val_las.write(output_path)
                        log.info(f"Saved predictions to {output_path}")

                    elem_filepath = batch.filepath[sample_idx]
                    self.in_memory_tile_id = elem_tile_id
                    self.val_las = laspy.read(elem_filepath)
                    # param = laspy.ExtraBytesParams(name="building_proba", type=float)
                    # self.val_las.add_extra_dim(param)
                    # TODO: consider setting this to np.nan or equivalent to capture incomplete predictions.
                    self.val_las.classification[:] = 0
                    self.val_las.gps_time[:] = 0.0
                    self.val_las_pos = np.asarray(
                        [
                            self.val_las.x,
                            self.val_las.y,
                            self.val_las.z,
                        ],
                        dtype=np.float32,
                    ).transpose()
                    self.val_las_pos = torch.from_numpy(self.val_las_pos)

                elem_preds = preds[batch.batch == sample_idx]
                elem_proba = proba[batch.batch == sample_idx][:, 1]
                elem_pos = batch.origin_pos[batch.batch == sample_idx]
                assign_idx = knn(self.val_las_pos, elem_pos, k=1, num_workers=1)[1]
                self.val_las.classification[assign_idx] = elem_preds
                # TODO: remove this ugly hack that sets predictions as gps_time !!
                self.val_las.gps_time[assign_idx] = elem_proba

    def on_validation_end(self):
        """Save the last unsaved predicted las."""
        output_path = osp.join(
            self.val_preds_folder,
            f"{self.in_memory_tile_id}.las",
        )
        self.val_las.write(output_path)

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, _, preds, targets = self.step(batch)

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
