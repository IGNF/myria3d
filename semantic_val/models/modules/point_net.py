import torch
from torch import nn
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq


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

        self.input_cloud_size = hparams["input_cloud_size"]
        self.mlp1 = MLP(hparams["MLP1_channels"])
        self.mlp2 = MLP(hparams["MLP2_channels"])
        self.mlp3 = MLP(hparams["MLP3_channels"])

        self.softmax = nn.Softmax(dim=2)

    def forward(self, input):
        """
        the forward function producing the embeddings for each point of 'input'
        input = [n_batch, input_feat, subsample_size] float array: input features
        output = [n_batch,n_class, subsample_size] float array: point class logits
        """
        f1 = self.mlp1(input)
        f2 = self.mlp2(f1)
        context_vector = torch.max(f2, 1)[0]
        input_size = f1.shape[1]
        expanded_context_vector = torch.unsqueeze(context_vector, 1).expand(-1, input_size, -1)
        Gf1 = torch.cat((expanded_context_vector, f1), 2)
        scores = self.mlp3(Gf1)
        logits = self.softmax(scores)

        return logits
