# Adapted from https://github.com/aRI0U/RandLA-Net-pytorch/blob/master/model.py

import torch
import torch.nn as nn

from torch_points_kernels import knn


class RandLANet(nn.Module):
    def __init__(self, hparams_net: dict):
        """
        Implementation follows original paper: RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds
        https://arxiv.org/abs/1911.11236

        Our modifications:
        - fc_start = nn.Linear(d_in, d_in * 2) instead of self.fc_start = nn.Linear(d_in, 8) to avoid
        information bottleneck in cases where d_in is above 8.
        """
        super(RandLANet, self).__init__()
        self.d_in = hparams_net.get("d_in", 6)  # xyz + features
        self.num_neighbors = hparams_net.get("num_neighbors", 16)
        self.decimation = hparams_net.get("decimation", 4)
        self.dropout = hparams_net.get("dropout", 0.0)
        self.num_classes = hparams_net.get("num_classes", 6)

        self.fc_start = nn.Linear(self.d_in, self.d_in * 2)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(self.d_in * 2, eps=1e-6, momentum=0.99), nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList(
            [
                LocalFeatureAggregation(self.d_in * 2, 16, self.num_neighbors),
                LocalFeatureAggregation(32, 64, self.num_neighbors),
                LocalFeatureAggregation(128, 128, self.num_neighbors),
                LocalFeatureAggregation(256, 256, self.num_neighbors),
            ]
        )

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(transpose=True, bn=True, activation_fn=nn.ReLU())
        self.decoder = nn.ModuleList(
            [
                SharedMLP(1024, 256, **decoder_kwargs),
                SharedMLP(512, 128, **decoder_kwargs),
                SharedMLP(256, 32, **decoder_kwargs),
                SharedMLP(64, self.d_in * 2, **decoder_kwargs),
            ]
        )
        self.set_fc_end(self.d_in, self.dropout, self.num_classes)

    def set_fc_end(self, d_in, dropout, num_classes):
        """Build the final fully connected layer.

        Args:
            d_in (int): number of input features
            dropout (float): dropout level in final FC layer
            num_classes (int): number of output classes
        """
        parts = [
            SharedMLP(d_in * 2, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
        ]
        if dropout:
            parts.append(nn.Dropout(p=dropout))
        parts.append(SharedMLP(32, num_classes))
        self.fc_end = nn.Sequential(*parts)

    def forward(self, batch):
        """, shape


        Args:
            batch (pytorch_geometric.Data): Subtile information with shape (B*N, 3+F).
            Attributs: pos (B*N, 3) and x (B*N, F) which contains cloud XYZ positions and features.

        Returns:
            torch.Tensor: classification logits for each point, with shape (B*num_classes,C)
        """

        input = torch.cat([batch.pos, batch.x], axis=1)
        chunks = torch.split(input, len(batch.pos) // batch.batch_size)
        input = torch.stack(chunks)  # B, N, 3+F

        N = input.size(1)
        d = self.decimation

        coords = input[..., :3].clone()  # .cpu()
        x = self.fc_start(input).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:, : N // decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:, :, : N // decimation_ratio]

        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:, : N // decimation_ratio].cpu().contiguous(),  # original set
                coords[:, : d * N // decimation_ratio]
                .cpu()
                .contiguous(),  # upsampled set
                1,
            )  # shape (B, N, 1)
            neighbors = neighbors.to(x.device)
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:, :, torch.argsort(permutation)]

        scores = self.fc_end(x)

        scores = scores.squeeze(-1)  # B, C, N
        scores = torch.cat(
            [score_cloud.permute(1, 0) for score_cloud in scores]
        )  # B*N, C
        return scores  # B*N, C

    def change_num_class_for_finetuning(self, new_num_classes: int):
        """
        Change end layer output number of classes if new_num_classes is different.
        This method is used for finetuning.

        Args:
            new_num_classes (int): new number of classes for finetuning pretrained model.
        """
        if new_num_classes != self.num_classes:
            self.fc_end[-1] = SharedMLP(32, new_num_classes)
            self.num_classes = new_num_classes


class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode="zeros",
        bn=False,
        activation_fn=None,
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode,
        )
        self.batch_norm = (
            nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        )
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
        Forward pass of the network
        Parameters
        ----------
        input: torch.Tensor, shape (B, d_in, N, K)
        Returns
        -------
        torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        r"""
        Forward pass
        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d, N, 1)
            features of the point cloud
        neighbors: tuple
        Returns
        -------
        torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        idx = idx.to(coords.device)
        dist = dist.to(coords.device)
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)
        neighbors = neighbors.to(coords.device)

        # relative point position encoding
        concat = torch.cat(
            (
                extended_coords,
                neighbors,
                extended_coords - neighbors,
                dist.unsqueeze(-3),
            ),
            dim=-3,
        )
        concat = concat.to(coords.device)
        return torch.cat((self.mlp(concat), features.expand(B, -1, N, K)), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False), nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(
            in_channels, out_channels, bn=True, activation_fn=nn.ReLU()
        )

    def forward(self, x):
        r"""
        Forward pass
        Parameters
        ----------
        x: torch.Tensor, shape (B, d_in, N, K)
        Returns
        -------
        torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)

        return self.mlp(features)


class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2 * d_out)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out // 2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out // 2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
        Forward pass
        Parameters
        ----------
        coords: torch.Tensor, shape (B, N, 3)
            coordinates of the point cloud
        features: torch.Tensor, shape (B, d_in, N, 1)
            features of the point cloud
        Returns
        -------
        torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        # torch_geometric KNN supports CUDA but would need a batch_x and batch_y index tensor.
        knn_output = knn(
            coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors
        )
        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))
