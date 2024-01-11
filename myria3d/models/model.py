import os
from pathlib import Path
import tempfile
from typing import Optional
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn.pool import global_mean_pool
from myria3d.metrics.confusion_matrix import save_confusion_matrix
from myria3d.metrics.polygon_metrics import make_polygon_metrics

from myria3d.models.modules.pyg_randla_net import PyGRandLANet
from myria3d.metrics import ConfusionMatrix
from myria3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [PyGRandLANet]
import pandas as pd

PREDICTION_FILE = Path(os.getcwd()) / "./predictions.csv"


def get_neural_net_class(class_name: str) -> nn.Module:
    """A Class Factory to class of neural net based on class name.

    :meta private:

    Args:
        class_name (str): the name of the class to get.

    Returns:
        nn.Module: CLass of requested neural network.
    """
    for neural_net_class in MODEL_ZOO:
        if class_name in neural_net_class.__name__:
            return neural_net_class
    raise KeyError(f"Unknown class name {class_name}")


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

    def __init__(self, **kwargs):
        """Initialization method of the Model lightning module.

        Everything needed to train/test/predict with a neural architecture, including
        the architecture class name and its hyperparameter.

        See config files for a list of kwargs.

        """
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        neural_net_class = get_neural_net_class(self.hparams.neural_net_class_name)
        self.model = neural_net_class(**self.hparams.neural_net_hparams)

        self.softmax = nn.Softmax(dim=1)
        self.criterion = self.hparams.criterion
        self.class_names = [name for name in self.hparams.classification_dict.values()]

    def setup(self, stage: Optional[str]) -> None:
        """Setup stage: prepare to compute IoU and loss."""
        num_classes = self.hparams.num_classes
        if stage == "fit":
            self.train_iou = self.hparams.iou()
            self.val_iou = self.hparams.iou()
            self.train_cm = ConfusionMatrix(num_classes)
            self.val_cm = ConfusionMatrix(num_classes)
        if stage == "test":
            self.test_iou = self.hparams.iou()
            self.test_cm = ConfusionMatrix(num_classes)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass of neural network.

        Args:
            batch (Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            torch.Tensor (B*N,1): targets
            torch.Tensor (B*N,C): logits

        """
        logits = self.model(batch.x, batch.pos, batch.batch, batch.ptr)
        if self.training or "copies" not in batch:
            # In training mode and for validation, we directly optimize on subsampled points, for
            # 1) Speed of training - because interpolation multiplies a step duration by a 5-10 factor!
            # 2) data augmentation at the supervision level.
            logits_classification = global_mean_pool(logits, batch.batch)
            return batch.y, logits_classification  # B*N, C

        # During evaluation on test data and inference, we interpolate predictions back to original positions
        # KNN is way faster on CPU than on GPU by a 3 to 4 factor.
        logits = logits.cpu()
        batch_y = self._get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)
        logits = knn_interpolate(
            logits.cpu(),
            batch.copies["pos_sampled_copy"].cpu(),
            batch.copies["pos_copy"].cpu(),
            batch_x=batch.batch.cpu(),
            batch_y=batch_y.cpu(),
            k=self.hparams.interpolation_k,
            num_workers=self.hparams.num_workers,
        )
        targets = None  # no targets in inference mode.
        if "transformed_y_copy" in batch.copies:
            # eval (test/val).
            targets = batch.copies["transformed_y_copy"].to(logits.device)
        logits_classification = global_mean_pool(logits, batch_y)
        return targets, logits_classification

    def on_fit_start(self) -> None:
        """On fit start: get the experiment for easier access."""
        self.experiment = self.logger.experiment[0]
        self.criterion = self.criterion.to(self.device)

    def on_test_start(self) -> None:
        """On test start: get the experiment for easier access."""
        self.experiment = self.logger.experiment[0]
        self.criterion = self.criterion.to(self.device)

    def on_train_start(self):
        # By default, lightning executes validation step sanity checks
        # before training starts, so we need to make sure val cm does not store it.
        self.val_cm.reset()

    def training_step(self, batch: Batch, batch_idx: int) -> dict:
        """Training step.

        Makes a model pass. Then, computes loss and predicted class of subsampled points to log loss and IoU.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.
            batch_idx (int): batch identified (unused)

        Returns:
            dict: a dict containing the loss, logits, and targets.
        """
        targets, logits = self.forward(batch)
        self.criterion = self.criterion.to(logits.device)
        self.train_cm = self.train_cm.to(logits.device)
        loss = self.criterion(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        with torch.no_grad():
            preds = torch.argmax(logits.detach(), dim=1)
            self.train_iou(preds, targets)
            self.train_cm(preds, targets)

        self.log(
            "train/iou",
            self.train_iou,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return {"loss": loss, "logits": logits, "targets": targets}

    def on_train_epoch_end(self):
        # `outputs` is a list of dicts returned from `training_step()`
        self.log("train/miou", self.train_cm.miou(), prog_bar=True)
        self.log("train/oa", self.train_cm.oa(), prog_bar=True)
        self.log("train/macc", self.train_cm.macc(), prog_bar=True)
        # TODO: find class_names.
        # for iou, seen, name in zip(*self.train_cm.iou(), self.class_names):
        #     if seen:
        #         self.log(f"train/iou_{name}", iou, prog_bar=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_precision, path_recall = save_confusion_matrix(
                self.train_cm.confmat, tmpdir, self.class_names
            )
            self.experiment.log_image(path_precision, name="Train Precision CM")
            self.experiment.log_image(path_recall, name="Train Recall CM")

        # Does not work, we do not know why... Maybe due to debug mode
        self.log_all_cms(phase="Train", cm_object=self.train_cm)
        self.train_cm.reset()

    def log_all_cms(self, phase: str, cm_object: ConfusionMatrix):
        self.experiment.log_confusion_matrix(
            matrix=cm_object.confmat.cpu().numpy().tolist(),
            labels=self.class_names,
            file_name=f"{phase} CM",
            title="{phase} Confusion Matrix",
            epoch=self.current_epoch,
        )
        self.experiment.log_confusion_matrix(
            matrix=cm_object.get_pinus_cm(self.class_names).tolist(),
            labels=["Non-Pinus", "Pinus"],
            file_name=f"{phase} CM Pinus",
            title="{phase} Confusion Matrix Pinus",
            epoch=self.current_epoch,
        )
        self.experiment.log_confusion_matrix(
            matrix=cm_object.get_quercus_cm(self.class_names).tolist(),
            labels=["Non-Quercus", "Quercus"],
            file_name=f"{phase} CM Quercus",
            title="{phase} Confusion Matrix Quercus",
            epoch=self.current_epoch,
        )
        self.experiment.log_confusion_matrix(
            matrix=cm_object.get_needleleaf_cm(self.class_names).tolist(),
            labels=["Non-Needleleaf", "Needleleaf"],
            file_name=f"{phase} CM Needleleaf",
            title="{phase} Confusion Matrix Needleleaf",
            epoch=self.current_epoch,
        )

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        """Validation step.

        Makes a model pass. Then, computes loss and predicted class of subsampled points to log loss and IoU.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.
            batch_idx (int): batch identified (unused)

        Returns:
            dict: a dict containing the loss, logits, and targets.

        """
        targets, logits = self.forward(batch)
        self.criterion = self.criterion.to(logits.device)
        self.val_cm = self.val_cm.to(logits.device)
        loss = self.criterion(logits, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True)

        preds = torch.argmax(logits.detach(), dim=1)
        self.val_iou = self.val_iou.to(preds.device)
        self.val_iou(preds, targets)
        self.val_cm(preds, targets)

        self.log("val/iou", self.val_iou, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss, "logits": logits, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        """At the end of a validation epoch, compute the IoU.

        Args:
            outputs : output of validation_step

        """
        self.val_iou.compute()

        miou = self.val_cm.miou()
        oa = self.val_cm.oa()
        macc = self.val_cm.macc()

        self.log("val/miou", miou, prog_bar=True)
        self.log("val/oa", oa, prog_bar=True)
        self.log("val/macc", macc, prog_bar=True)
        # for iou, seen, name in zip(*self.val_cm.iou(), self.class_names):
        #     if seen:
        #         self.log(f"val/iou_{name}", iou, prog_bar=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_precision, path_recall = save_confusion_matrix(
                self.val_cm.confmat, tmpdir, self.class_names
            )
            self.experiment.log_image(path_precision, name="Val Precision CM")
            self.experiment.log_image(path_recall, name="Val Recall CM")

        if hasattr(self, "experiment"):
            self.log_all_cms(phase="Val", cm_object=self.val_cm)
        self.val_cm.reset()

    def test_step(self, batch: Batch, batch_idx: int):
        """Test step.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            dict: Dictionnary with full-cloud predicted logits as well as the full-cloud (transformed) targets.

        """
        targets, logits = self.forward(batch)
        self.criterion = self.criterion.to(logits.device)
        self.test_cm = self.test_cm.to(logits.device)
        loss = self.criterion(logits, targets)
        self.log("test/loss", loss, on_step=True, on_epoch=True)

        preds = torch.argmax(logits, dim=1)
        self.test_iou = self.test_iou.to(preds.device)
        self.test_iou(preds, targets)
        if targets is not None:
            self.test_cm(preds, targets)

        self.log(
            "test/iou",
            self.test_iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        df = pd.DataFrame(
            # need to remove prefix
            data={
                "patch_stem": batch.patch_id,
                "preds": preds.cpu().numpy(),
                "targets": targets.cpu().numpy(),
            }
        )
        hdr = False if os.path.isfile(PREDICTION_FILE) else True
        df.to_csv(PREDICTION_FILE, index=False, mode="a", header=hdr)

        return {"loss": loss, "logits": logits, "targets": targets}

    def on_test_epoch_end(self):
        # `outputs` is a list of dicts returned from `test_step()`
        self.log("test/miou", self.test_cm.miou(), prog_bar=True)
        self.log("test/oa", self.test_cm.oa(), prog_bar=True)
        self.log("test/macc", self.test_cm.macc(), prog_bar=True)
        # for iou, seen, name in zip(*self.test_cm.iou(), self.class_names):
        #     if seen:
        #         self.log(f"test/iou_{name}", iou, prog_bar=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_precision, path_recall = save_confusion_matrix(
                self.test_cm.confmat, tmpdir, self.class_names
            )
            self.experiment.log_image(path_precision, name="Test Precision CM")
            self.experiment.log_image(path_recall, name="Test Recall CM")

        if hasattr(self, "experiment"):
            self.log_all_cms(phase="Test", cm_object=self.test_cm)
            cm_polygon_path = make_polygon_metrics(PREDICTION_FILE)
            self.experiment.log_image(cm_polygon_path, name="Polygon CM")

        self.test_cm.reset()

    def predict_step(self, batch: Batch) -> dict:
        """Prediction step.

        Move to CPU to avoid acucmulation of predictions into gpu memory.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            dict: Dictionnary with predicted logits as well as input batch.

        """
        _, logits = self.forward(batch)
        return {"logits": logits.detach().cpu()}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            An optimizer, or a config of a scheduler and an optimizer.

        """
        self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.hparams.lr_scheduler(optimizer),
            "monitor": self.hparams.monitor,
        }

    def _get_batch_tensor_by_enumeration(self, pos_x: torch.Tensor) -> torch.Tensor:
        """Get batch tensor (e.g. [0,0,1,1,2,2,...,B-1,B-1] )
        from shape B,N,... to shape (N,...).
        """
        return torch.cat([torch.full((len(sample_pos),), i) for i, sample_pos in enumerate(pos_x)])
