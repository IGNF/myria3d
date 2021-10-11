import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn.glob.glob import global_max_pool

from semantic_val.datamodules.datasets.lidar_transforms import get_subsampling_mask


def MLP(channels, batch_norm: bool = True):
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

        self.mlp1 = MLP(hparams["MLP1_channels"], batch_norm=hparams["batch_norm"])
        self.mlp2 = MLP(hparams["MLP2_channels"], batch_norm=hparams["batch_norm"])
        self.mlp3 = MLP(hparams["MLP3_channels"], batch_norm=hparams["batch_norm"])
        self.lin = Lin(hparams["MLP3_channels"][-1], hparams["num_classes"])

    def forward(self, batch):
        """
        Object batch is a PyG data.Batch, as defined in custom collate_fn.
        Tensors pos and x (features) are in long format (B*N, M) expected by pyG methods.
        """
        features = torch.cat([batch.pos, batch.x], axis=1)
        input_size = features.shape[0]
        subsampling_size = (batch.batch_x == 0).sum()

        # Pass through network layers
        f1 = self.mlp1(features)
        f2 = self.mlp2(f1)
        context_vector = global_max_pool(f2, batch.batch_x)
        expanded_context_vector = (
            context_vector.unsqueeze(1)
            .expand((-1, subsampling_size, -1))
            .reshape(input_size, -1)
        )
        Gf1 = torch.cat((expanded_context_vector, f1), 1)
        f3 = self.mlp3(Gf1)
        logits = self.lin(f3)

        return logits
