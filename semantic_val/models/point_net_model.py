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
from torch_geometric.nn.unpool.knn_interpolate import knn_interpolate

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

    :param save_predictions: Set to True to save LAS files with predictions as classification field.
    Only in effect if save_predictions is True.
    """

    def __init__(
        self,
        num_classes: int = 2,
        MLP1_channels: List[int] = [6, 32, 32],
        MLP2_channels: List[int] = [32, 64, 128],
        MLP3_channels: List[int] = [160, 128, 64, 32],
        batch_norm: bool = False,
        lr: float = 0.01,
        save_predictions: bool = False,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        self.model = Net(hparams=self.hparams)
        self.save_predictions = save_predictions
        self.in_memory_tile_id = ""

        self.softmax = nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 1.0]))

        self.train_iou = IoU(num_classes, reduction="none")
        self.val_iou = IoU(num_classes, reduction="none")
        self.test_iou = IoU(num_classes, reduction="none")
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

        self.max_reached_val_iou = -np.inf
        self.val_iou_accumulator = []

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
        self.val_preds_folder = osp.join(log_path, "validation_preds")
        os.makedirs(self.val_preds_folder, exist_ok=True)
        self.val_preds_geotiffs_folder = osp.join(self.val_preds_folder, "geotiffs")
        os.makedirs(self.val_preds_geotiffs_folder, exist_ok=True)

        self.experiment = self.logger.experiment[0]
        self.experiment.log_parameter("experiment_logs_dirpath", log_path)

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, preds, targets = self.step(batch)

        acc = self.train_accuracy(preds, targets)
        iou = self.train_iou(preds, targets)[1]
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/iou", iou, on_step=True, on_epoch=True, prog_bar=True)

        preds_avg = (preds * 1.0).mean().item()
        targets_avg = (targets * 1.0).mean().item()
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

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, _, proba, preds, targets = self.step(batch)

        acc = self.val_accuracy(preds, targets)
        iou = self.val_iou(preds, targets)[1]
        self.val_iou_accumulator.append(iou)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/iou", iou, on_step=True, on_epoch=True, prog_bar=True)

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
            "preds": preds,
            "targets": targets,
            "batch": batch,
        }

    def save_val_las(self):
        """After inference of classification in self.val_las, save:
        - The LAS with updated classification
        - A GeoTIFF of the classification, which is logged into Comet as well.
        """
        output_path = osp.join(
            self.val_preds_folder,
            f"{self.in_memory_tile_id}.las",
        )
        self.val_las.write(output_path)
        log.info(f"Saved predictions to {output_path}")

        # Create the tiff to self.val_preds_geotiffs_folder + name
        output_path = osp.join(
            self.val_preds_geotiffs_folder,
            f"{self.in_memory_tile_id}.tif",
        )

    def validation_step_end(self, output: dict):
        """Save the predicted classes in las format with position."""

        if self.save_predictions:
            proba = output["proba"]
            preds = output["preds"]
            batch = output["batch"]
            targets = output["targets"]
            for sample_idx in range(len(np.unique(batch.batch))):
                elem_tile_id = batch.tile_id[sample_idx]
                if self.in_memory_tile_id != elem_tile_id:
                    if self.in_memory_tile_id:
                        self.save_val_las()
                    self.in_memory_tile_id = elem_tile_id
                    self.val_las = laspy.read(batch.filepath[sample_idx])
                    param = laspy.ExtraBytesParams(name="building_proba", type=float)
                    self.val_las.add_extra_dim(param)
                    param = laspy.ExtraBytesParams(
                        name="classification_confusion", type=int
                    )
                    self.val_las.add_extra_dim(param)

                    # TODO: consider setting this to np.nan or equivalent to capture incomplete predictions.
                    self.val_las.classification[:] = 0
                    self.val_las_pos = np.asarray(
                        [
                            self.val_las.x,
                            self.val_las.y,
                            self.val_las.z,
                        ],
                        dtype=np.float32,
                    ).transpose()
                    self.val_las_pos = torch.from_numpy(self.val_las_pos)

                elem_pos = batch.pos_copy[batch.batch_y == sample_idx]
                elem_preds = preds[batch.batch_y == sample_idx]
                elem_proba = proba[batch.batch_y == sample_idx][:, 1]
                elem_targets = targets[batch.batch_y == sample_idx]

                assign_idx = knn(self.val_las_pos, elem_pos, k=1, num_workers=1)[1]

                self.val_las.classification[assign_idx] = elem_preds
                elem_predsdiff = elem_preds + 2 * elem_targets
                self.val_las.classification_confusion[assign_idx] = elem_predsdiff
                self.val_las.building_proba[assign_idx] = elem_proba

    def on_validation_end(self):
        """Save the last unsaved predicted las and keep track of best IoU"""
        output_path = osp.join(
            self.val_preds_folder,
            f"{self.in_memory_tile_id}.las",
        )
        self.val_las.write(output_path)

        val_iou = np.mean(self.val_iou_accumulator)
        self.max_reached_val_iou = max(val_iou, self.max_reached_val_iou)
        self.experiment.log_metric("val/max_iou", self.max_reached_val_iou)

    def test_step(self, batch: Any, batch_idx: int):
        loss, _, _, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_accuracy(preds, targets)
        iou = self.test_iou(preds, targets)[1]

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        self.log("test/iou", iou, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

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
