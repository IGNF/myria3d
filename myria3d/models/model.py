from typing import Optional
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torchmetrics import MaxMetric
from myria3d.models.modules.point_net2 import PointNet2
from myria3d.models.modules.randla_net import RandLANet
from myria3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [RandLANet, PointNet2]


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
        self.model = neural_net_class(self.hparams.neural_net_hparams)

        self.softmax = nn.Softmax(dim=1)
        # TODO: This should be uncommented for prediction of finetuned model
        # but not for finetuning itself... To be investigated!
        self.criterion = self.hparams.criterion

    def setup(self, stage: Optional[str]) -> None:
        """Setup stage: prepare to compute IoU and loss."""
        if stage == "fit":
            self.train_iou = self.hparams.iou()
            self.val_iou = self.hparams.iou()
            self.val_iou_best = MaxMetric()
        if stage == "test":
            self.test_iou = self.hparams.iou()
        if stage != "predict":
            self.criterion = self.hparams.criterion

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward pass of neural network.

        Args:
            batch (Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            torch.Tensor (B*N,C): logits

        """
        # TODO: replace with call to encoder + regression !
        return torch.cat(
            [batch.obbox_dict[k][:, None] for k in ["Ax", "Ay", "Bx", "By", "D"]],
            dim=-1,
        ).requires_grad_()

    def step(self, batch: Batch):
        """Model step, including loss computation.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            torch.Tensor (1), torch.Tensor (B*N,C), torch.Tensor (B*N,C), torch.Tensor: loss, logits, targets

        """
        predicted_obbox = self.forward(batch)
        target_obbox = torch.cat(
            [batch.obbox_dict[k][:, None] for k in ["Ax", "Ay", "Bx", "By", "D"]],
            dim=-1,
        )
        loss = self.criterion(predicted_obbox, target_obbox)
        return loss

    def on_fit_start(self) -> None:
        """On fit start: get the experiment for easier access."""
        self.experiment = self.logger.experiment[0]

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
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

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
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def test_step(self, batch: Batch, batch_idx: int):
        """Test step.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            dict: Dictionnary with full-cloud predicted logits as well as the full-cloud (transformed) targets.

        """
        loss = self.step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def predict_step(self, batch: Batch) -> dict:
        """Prediction step.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            dict: Dictionnary with predicted logits as well as input batch.

        """
        predicted_obbox = self.forward(batch)
        return {"predicted_obbox": predicted_obbox.detach().cpu()}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Returns:
            An optimized, or a config which includes a scheduler and the parma to monitor.

        """
        self.lr = self.hparams.lr  # aliasing for Lightning auto_find_lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        try:
            lr_scheduler = self.hparams.lr_scheduler(optimizer)
        except Exception:
            # OneCycleLR needs optimizer and max_lr
            lr_scheduler = self.hparams.lr_scheduler(optimizer, self.lr)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": self.hparams.monitor,
        }

        return config
