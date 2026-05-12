import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import knn_interpolate

from myria3d.models.modules.pyg_randla_net import PyGRandLANet
from myria3d.models.modules.pointnet import PointNet
from myria3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO = [PyGRandLANet, PointNet]


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
    """Model training, validation, test and prediction of point cloud semantic segmentation.

    During training and validation, metrics are calculed based on sumbsampled points only.
    At test time, metrics are calculated considering all the points.

    To keep this module light, a callback takes care of metric computations.


    Read the Pytorch Lightning docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

    """

    def __init__(self, **kwargs):
        """Initialization method of the Model lightning module.

        Everything needed to train/evaluate/test/predict with a neural architecture, including
        the architecture class name and its hyperparameter.

        See config files for a list of kwargs.

        """
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(ignore=["criterion"])

        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.model = neural_net_class(**kwargs.get("neural_net_hparams"))

        self.softmax = nn.Softmax(dim=1)
        self.criterion = kwargs.get("criterion")

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
            return batch.y, logits  # B*N, C

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
        return targets, logits

    def training_step(self, batch: Batch, batch_idx: int) -> dict:
        """Training step.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.
            batch_idx (int): batch identified (unused)

        Returns:
            dict: a dict containing the loss, logits, and targets.
        """
        targets, logits = self.forward(batch)
        self.criterion = self.criterion.to(logits.device)
        loss = self.criterion(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss, "logits": logits, "targets": targets}

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        """Validation step.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.
            batch_idx (int): batch identified (unused)

        Returns:
            dict: a dict containing the loss, logits, and targets.

        """
        targets, logits = self.forward(batch)
        self.criterion = self.criterion.to(logits.device)
        loss = self.criterion(logits, targets)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "logits": logits, "targets": targets}

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
        loss = self.criterion(logits, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "logits": logits, "targets": targets}

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
