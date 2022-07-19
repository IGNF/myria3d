import os
from typing import Optional
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torchmetrics import MaxMetric
from myria3d.models.modules.point_net2 import PointNet2
from myria3d.models.modules.randla_net import RandLANet
from myria3d.utils import utils
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import geopandas as gpd

log = utils.get_logger(__name__)

MODEL_ZOO = [RandLANet, PointNet2]
SUBTILE_WIDTH = 50.0
BIAS_OF_LOGITS = torch.Tensor(
    [-0.5, -0.5, -0.5, -0.5, 0.0]
)  # to shift to relative xy and keep absolute width between 0 and 1
SCALING_OF_LOGITS = torch.Tensor([SUBTILE_WIDTH for _ in range(5)])


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
        # self.bbox_layer_mlp2 = SharedMLP(512, 5, activation_fn=nn.ReLU(), bn=True)

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
        input = torch.cat([batch.pos, batch.x], axis=1)
        chunks = torch.split(input, len(batch.pos) // batch.num_graphs)
        input = torch.stack(chunks)  # B, N, 3+F

        N = input.size(1)

        pos = input[..., :3].clone()
        x = self.model.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.model.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        pos = pos[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.model.encoder:
            # at iteration i, x.shape = (B, N//(self.decimation**i), d_in)
            x = lfa(pos[:, : N // decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= self.model.decimation
            x = x[:, :, : N // decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x = self.model.mlp(x)  # B, ~48, 512
        # outputs sigmoid activated values.
        x = torch.max(x, dim=-2).values.squeeze()  # B, 512

        x = self.model.regression_layer(x)  # B, 1, 5+1 softmaxée
        scores = x[:, 5:6]
        x = x[:, :5] * scores

        predicted_bbox = (
            x
            + BIAS_OF_LOGITS.repeat(batch.num_graphs)
            .view([batch.num_graphs, -1])
            .to(x.device)
        ) * SCALING_OF_LOGITS.repeat(batch.num_graphs).view([batch.num_graphs, -1]).to(
            x.device
        )

        return predicted_bbox

    def step(self, batch: Batch):
        """Model step, including loss computation.

        Args:
            batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
            and y (targets, optionnal) in (B*N,C) format.

        Returns:
            torch.Tensor (1), torch.Tensor (B*N,C), torch.Tensor (B*N,C), torch.Tensor: loss, logits, targets

        """
        predicted_obbox = self.forward(batch)

        target_obbox_AB = torch.cat(
            [batch.obbox_dict[k][:, None] for k in ["Ax", "Ay", "Bx", "By", "D"]],
            dim=-1,
        )  # B, 5
        target_obbox_BA = torch.cat(
            [batch.obbox_dict[k][:, None] for k in ["Bx", "By", "Ax", "Ay", "D"]],
            dim=-1,
        )  # B, 5

        # Clamping car dépasse de la zone d'intérêt parfois
        target_obbox_AB[:, :4] = torch.clamp(
            target_obbox_AB[:, :4], min=-SUBTILE_WIDTH / 2.0, max=SUBTILE_WIDTH / 2.0
        )
        target_obbox_BA[:, :4] = torch.clamp(
            target_obbox_BA[:, :4], min=-SUBTILE_WIDTH / 2.0, max=SUBTILE_WIDTH / 2.0
        )

        # if not self.training:
        #     # eval mode -> need to ignore smaller bridges...
        #     pred_bridge_length = torch.sqrt(
        #         torch.square(predicted_obbox[:, :2] - predicted_obbox[:, 2:4]).sum(1)
        #     )
        #     MIN_BRIDGE_LENGTH = 3.5
        #     mask_too_small_bridges = (pred_bridge_length > MIN_BRIDGE_LENGTH).unsqueeze(
        #         1
        #     )
        #     predicted_obbox = predicted_obbox * mask_too_small_bridges

        # Invariance to order by taking the mean loss i.e. closest point.
        losses_AB = self.criterion(predicted_obbox, target_obbox_AB).cpu()
        losses_BA = self.criterion(predicted_obbox, target_obbox_BA).cpu()

        losses = []
        targets = []
        for idx in range(batch.num_graphs):
            ab_loss = losses_AB[idx]
            ba_loss = losses_BA[idx]
            loss = ab_loss
            target = target_obbox_AB[idx]
            if ab_loss.mean() > ba_loss.mean():
                loss = ba_loss
                target = target_obbox_BA[idx]
            losses += [loss.clone()]
            targets += [target.clone()]

        losses = torch.stack(losses)
        targets = torch.stack(targets)

        # means_by_sample = stacked_losses.mean(dim=1)  # 50,2
        # idx_of_minimizing_permutation = torch.argmin(means_by_sample, dim=1)  # B
        # torch.gather(stacked_losses, 2, idx_of_minimizing_permutation)
        # losses = torch.take_along_dim(
        #     stacked_losses, idx_of_minimizing_permutation, dim=1
        # )
        # stacked_targets = torch.stack([target_obbox_AB, target_obbox_BA])
        # target_obbox = torch.take_along_dim(
        #     stacked_targets, idx_of_minimizing_permutation
        # )

        log.info(
            f"Predictions Ax, Ay, Bx, By, D {predicted_obbox.mean(dim=0).cpu().detach()}"
        )
        return losses, predicted_obbox, targets

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
        losses, predicted_obbox, target_obbox = self.step(batch)

        if self.current_epoch % 5 == 0:
            d_target = pd.DataFrame(
                data=target_obbox.cpu().numpy(),
                columns=["Ax_train", "Ay_train", "Bx_train", "By_train", "D_train"],
            )
            d_pred = pd.DataFrame(
                data=predicted_obbox.detach().cpu().numpy(),
                columns=[
                    "Ax_pred_train",
                    "Ay_pred_train",
                    "Bx_pred_train",
                    "By_pred_train",
                    "D_pred_train",
                ],
            )
            d_loss = pd.DataFrame(
                data=(predicted_obbox - target_obbox).detach().cpu().numpy(),
                columns=[
                    "Ax_error_train",
                    "Ay_error_train",
                    "Bx_error_train",
                    "By_error_train",
                    "D_error_train",
                ],
            )
            df = d_target.join(d_pred).join(d_loss).astype(float).round(1)
            self.plot_and_log_from_df(df, phase="train")
            self.experiment.log_html(df.to_html(), clear=True)
        self.log_averages(losses, prefix="train/loss")
        self.log_averages(predicted_obbox, prefix="train/avg_pred")
        self.log_averages(target_obbox, prefix="train/avg_target")
        loss = losses.mean() / (SUBTILE_WIDTH / 2)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def plot_and_log_from_df(self, df: pd.DataFrame, phase="train"):
        for idx, row in df.iterrows():
            target = LineString(
                [
                    (row[f"Ax_{phase}"], row[f"Ay_{phase}"]),
                    (row[f"Bx_{phase}"], row[f"By_{phase}"]),
                ]
            ).buffer(
                row[f"D_{phase}"], cap_style=3
            )  # rectangular!
            pred = LineString(
                [
                    (row[f"Ax_pred_{phase}"], row[f"Ay_pred_{phase}"]),
                    (row[f"Bx_pred_{phase}"], row[f"By_pred_{phase}"]),
                ]
            ).buffer(
                row[f"D_pred_{phase}"], cap_style=3
            )  # rectangular!
            p = gpd.GeoSeries(target)
            ax = p.plot(color=["green"], alpha=0.5)
            p = gpd.GeoSeries(pred)
            p.plot(color=["blue"], alpha=0.5, ax=ax)
            ax.set_xlim([-SUBTILE_WIDTH / 2, SUBTILE_WIDTH / 2])
            ax.set_ylim([-SUBTILE_WIDTH / 2, SUBTILE_WIDTH / 2])
            plt.show()
            length = np.round(
                np.sqrt(
                    (row[f"Ax_{phase}"] - row[f"Bx_{phase}"]) ** 2
                    + (row[f"Ay_{phase}"] - row[f"By_{phase}"]) ** 2
                ),
                1,
            )
            path_to_save = (
                f"./diffplot_{phase}/{phase}_{row[f'D_{phase}']}_{length}.png"
            )
            # do not plot empty ones for now...
            if row[f"D_{phase}"] != 0:
                path_to_save = path_to_save.replace(".png", f"_{idx}.png")
                os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
                plt.savefig(path_to_save)
                self.experiment.log_image(path_to_save)

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
        losses, predicted_obbox, target_obbox = self.step(batch)
        d_target = pd.DataFrame(
            data=target_obbox.cpu().numpy(),
            columns=["Ax_val", "Ay_val", "Bx_val", "By_val", "D_val"],
        )
        d_pred = pd.DataFrame(
            data=predicted_obbox.cpu().numpy(),
            columns=[
                "Ax_pred_val",
                "Ay_pred_val",
                "Bx_pred_val",
                "By_pred_val",
                "D_pred_val",
            ],
        )
        d_loss = pd.DataFrame(
            data=(predicted_obbox - target_obbox).cpu().numpy(),
            columns=[
                "Ax_error_val",
                "Ay_error_val",
                "Bx_error_val",
                "By_error_val",
                "D_error_val",
            ],
        )
        df = d_target.join(d_pred).join(d_loss).astype(float).round(1)
        self.experiment.log_html(df.to_html())
        self.plot_and_log_from_df(df, phase="val")
        self.log_averages(losses, prefix="val/loss")
        self.log_averages(predicted_obbox, prefix="val/avg_pred")
        self.log_averages(target_obbox, prefix="val/avg_target")
        loss = losses.mean() / (SUBTILE_WIDTH / 2)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    # def test_step(self, batch: Batch, batch_idx: int):
    #     """Test step.

    #     Args:
    #         batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
    #         and y (targets, optionnal) in (B*N,C) format.

    #     Returns:
    #         dict: Dictionnary with full-cloud predicted logits as well as the full-cloud (transformed) targets.

    #     """
    #     loss = self.step(batch)
    #     self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
    #     return {"loss": loss}

    # def predict_step(self, batch: Batch) -> dict:
    #     """Prediction step.

    #     Args:
    #         batch (torch_geometric.data.Batch): Batch of data including x (features), pos (xyz positions),
    #         and y (targets, optionnal) in (B*N,C) format.

    #     Returns:
    #         dict: Dictionnary with predicted logits as well as input batch.

    #     """
    #     predicted_obbox = self.forward(batch)
    #     return {"predicted_obbox": predicted_obbox.detach().cpu()}

    def log_averages(self, tensor, prefix="train/loss"):
        with torch.no_grad():
            averages = tensor.mean(dim=0).cpu()
            self.log(
                f"{prefix}_Ax",
                averages[0],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}_Ay",
                averages[1],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}_Bx",
                averages[2],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}_By",
                averages[3],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                f"{prefix}_D",
                averages[4],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )

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
