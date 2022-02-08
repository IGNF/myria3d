import torch
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn.glob.glob import global_max_pool


class PointNet(nn.Module):
    def __init__(self, hparams_net: dict):
        super().__init__()

        self.num_classes = hparams_net["num_classes"]
        bn = hparams_net.get("batch_norm", True)
        d1 = hparams_net.get("MLP1_channels", [10, 64, 64])
        d2 = hparams_net.get("MLP2_channels", [64, 256, 512, 1024])
        d3 = hparams_net.get("MLP3_channels", [1088, 512, 256, 64, 4])

        self.mlp1 = MLP(d1, batch_norm=bn)
        self.mlp2 = MLP(d2, batch_norm=bn)
        self.mlp3 = MLP(d3, batch_norm=bn)
        self.lin = Linear(d3[-1], self.num_classes)

    def forward(self, batch):
        """
        Object batch is a PyG data.Batch, as defined in custom collate_fn.
        Tensors pos and x (features) are in long format (B*N, M) expected by pyG methods.
        """
        features = torch.cat([batch.pos, batch.x], axis=1)
        input_size = features.shape[0]
        subsampling_size = (batch.batch_x == 0).sum()

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

        self.change_num_class_for_finetuning(5)

        return logits

    def change_num_class_for_finetuning(self, new_num_classes: int):
        """
        Change end layer output number of classes if new_num_classes is different.
        This method is used for finetuning.

        Args:
            new_num_classes (int): new number of classes for finetuning pretrained model.
        """
        if new_num_classes != self.num_classes:
            mlp3_out_f = self.mlp3[-1][0].out_features
            self.lin = Linear(mlp3_out_f, new_num_classes)
            self.num_classes = new_num_classes


def MLP(channels, batch_norm: bool = True):
    return Seq(
        *[
            Seq(Linear(channels[i - 1], channels[i]), ReLU(), BatchNorm1d(channels[i]))
            if batch_norm
            else Seq(Linear(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ]
    )
