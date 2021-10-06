import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn.unpool.knn_interpolate import knn_interpolate

from semantic_val.datamodules.datasets.lidar_utils import get_subsampling_mask


def MLP(channels, batch_norm=False):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            if batch_norm
            else Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ]
    )


class PointNet(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.subsampling_size = hparams["subsampling_size"]
        self.mlp1 = MLP(hparams["MLP1_channels"])
        self.mlp2 = MLP(hparams["MLP2_channels"])
        self.mlp3 = MLP(hparams["MLP3_channels"])
        self.lin = Lin(hparams["MLP3_channels"][-1], 2)

    def forward(self, batch):
        """
        Object batch is a PyG data.Batch, with attr x, pos and y as well as batch (integer for assignment to sample).
        Format of x is (N1 + ... + Nk, C) which we convert to format (B * N, C) with N the subsampling_size.
        We use batch format (B, N, C) for nn logic, then go back to long format for KNN interpolation.
        """

        # Get back to batch shape
        x_list = []
        pos_list = []
        batch_x_list = []
        for sample_idx in range(len(np.unique(batch.batch))):
            x = batch.x[batch.batch == sample_idx]
            pos_x = batch.pos[batch.batch == sample_idx]
            batch_x = batch.batch[batch.batch == sample_idx]

            sampled_points_idx = get_subsampling_mask(len(x), self.subsampling_size)
            x = x[sampled_points_idx]
            pos_x = pos_x[sampled_points_idx]
            batch_x = batch_x[sampled_points_idx]

            x_list.append(x)
            pos_list.append(pos_x)
            batch_x_list.append(batch_x)
        pos = torch.stack(pos_list)
        x = torch.stack(x_list)
        features = torch.cat([pos, x], axis=2)

        # Pas through network layers
        f1 = self.mlp1(features)
        f2 = self.mlp2(f1)
        context_vector = torch.max(f2, 1)[0]
        input_size = f1.shape[1]
        expanded_context_vector = torch.unsqueeze(context_vector, 1).expand(
            -1, input_size, -1
        )
        Gf1 = torch.cat((expanded_context_vector, f1), 2)
        f3 = self.mlp3(Gf1)
        logits = self.lin(f3)

        # interpolate scores to all original points.
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=knn_interpolate#unpooling-layers
        logits = logits.view(-1, 2)  # (N_sub*B, C)
        pos_x = torch.cat(pos_list)
        pos_y = batch.pos
        batch_x = torch.cat(batch_x_list)
        batch_y = batch.batch
        logits = knn_interpolate(
            logits, pos_x, pos_y, batch_x=batch_x, batch_y=batch_y, k=3
        )
        return logits
