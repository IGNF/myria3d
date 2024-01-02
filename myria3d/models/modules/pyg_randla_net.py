import os.path as osp
from numbers import Number
from typing import Tuple
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch import LongTensor, Tensor
from torch.nn import Linear
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn_graph
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.utils import softmax
from torch_scatter import scatter, scatter_max
from torchmetrics.functional import jaccard_index
from tqdm import tqdm


class PyGRandLANet(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        decimation: int = 4,
        num_neighbors: int = 16,
        return_logits: bool = False,
    ):
        super().__init__()

        self.decimation = decimation
        # An option to return logits instead of probas
        self.return_logits = return_logits

        # Authors use 8, which is a bottleneck
        # for the final MLP, and also when num_classes>8
        # or num_features>8.
        d_bottleneck = max(32, num_classes, num_features)

        self.fc0 = Linear(num_features, d_bottleneck)
        self.block1 = DilatedResidualBlock(num_neighbors, d_bottleneck, 32)
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 256)
        self.block4 = DilatedResidualBlock(num_neighbors, 256, 512)
        self.mlp_summit = SharedMLP([512, 512])
        self.fp4 = FPModule(1, SharedMLP([512 + 256, 256]))
        self.fp3 = FPModule(1, SharedMLP([256 + 128, 128]))
        self.fp2 = FPModule(1, SharedMLP([128 + 32, 32]))
        self.fp1 = FPModule(1, SharedMLP([32 + 32, d_bottleneck]))
        self.mlp_classif = SharedMLP([d_bottleneck, 64, 32], dropout=[0.0, 0.5])
        self.fc_classif = Linear(32, num_classes)

    def forward(self, x, pos, batch, ptr, cluster_id):
        num_of_trees = scatter_max(cluster_id, batch)[0]
        cumsums = torch.concat(
            [torch.zeros(1, device=batch.device), torch.cumsum(num_of_trees, 0)]
        )
        cluster_id = [
            cluster_id[ptr[i] : ptr[i + 1]] + i + cumsums[i] for i in range(len(ptr) - 1)
        ]
        cluster_id = torch.concat(cluster_id)
        # Now we need to reorder everything !
        # The reordering will respect the batch limits so we do not need to reorder anything afterward.

        reordering = cluster_id.argsort()
        x = x[reordering]
        pos = pos[reordering]
        cluster_id = cluster_id[reordering]

        cluster_ptr = torch.concat(
            [
                torch.zeros(1, device=batch.device),
                torch.nonzero(cluster_id[:-1] != cluster_id[1:]).view(-1) + 1,
                torch.IntTensor([len(cluster_id)]).to(batch.device),
            ]
        ).int()

        b1_out = self.block1(self.fc0(x), pos, cluster_id)

        b1_out_decimated, ptr1 = decimate(b1_out, cluster_ptr, self.decimation)

        b2_out = self.block2(*b1_out_decimated)
        b2_out_decimated, ptr2 = decimate(b2_out, ptr1, self.decimation)

        b3_out = self.block3(*b2_out_decimated)
        b3_out_decimated, ptr3 = decimate(b3_out, ptr2, self.decimation)

        b4_out = self.block4(*b3_out_decimated)
        b4_out_decimated, _ = decimate(b4_out, ptr3, self.decimation)

        mlp_out = (
            self.mlp_summit(b4_out_decimated[0]),
            b4_out_decimated[1],
            b4_out_decimated[2],
        )

        fp4_out = self.fp4(*mlp_out, *b3_out_decimated)
        fp3_out = self.fp3(*fp4_out, *b2_out_decimated)
        fp2_out = self.fp2(*fp3_out, *b1_out_decimated)
        fp1_out = self.fp1(*fp2_out, *b1_out)

        x = self.mlp_classif(fp1_out[0])
        logits = self.fc_classif(x)

        if self.return_logits:
            return logits

        probas = logits.log_softmax(dim=-1)
        return probas


# Default activation, BatchNorm, and resulting MLP used by RandLA-Net authors
lrelu02_kwargs = {"negative_slope": 0.2}

bn099_kwargs = {"momentum": 0.01, "eps": 1e-6}


class SharedMLP(MLP):
    """SharedMLP following RandLA-Net paper."""

    def __init__(self, *args, **kwargs):
        # BN + Act always active even at last layer.
        kwargs["plain_last"] = False
        # LeakyRelu with 0.2 slope by default.
        kwargs["act"] = kwargs.get("act", "LeakyReLU")
        kwargs["act_kwargs"] = kwargs.get("act_kwargs", lrelu02_kwargs)
        # BatchNorm with 1 - 0.99 = 0.01 momentum
        # and 1e-6 eps by defaut (tensorflow momentum != pytorch momentum)
        kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", bn099_kwargs)
        super().__init__(*args, **kwargs)


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, channels):
        super().__init__(aggr="add")
        self.mlp_encoder = SharedMLP([10, channels // 2])
        self.mlp_attention = SharedMLP([channels, channels], bias=False, act=None, norm=None)
        self.mlp_post_attention = SharedMLP([channels, channels])

    def forward(self, edge_index, x, pos):
        out = self.propagate(edge_index, x=x, pos=pos)  # N, d_out
        out = self.mlp_post_attention(out)  # N, d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, index: Tensor) -> Tensor:
        """Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)
            index (Tensor): index of centroid positions
                (e.g. [0,...,0,1,...,1,...,N,...,N])

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        # Encode local neighboorhod structural information
        pos_diff = pos_j - pos_i
        distance = torch.sqrt((pos_diff * pos_diff).sum(1, keepdim=True))
        relative_infos = torch.cat([pos_i, pos_j, pos_diff, distance], dim=1)  # N * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N * K, d
        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N * K, 2d

        # Attention will weight the different features of x
        # along the neighborhood dimension.
        att_features = self.mlp_attention(local_features)  # N * K, d_out
        att_scores = softmax(att_features, index=index)  # N * K, d_out

        return att_scores * local_features  # N * K, d_out


class DilatedResidualBlock(torch.nn.Module):
    def __init__(
        self,
        num_neighbors,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = SharedMLP([d_in, d_out // 8])
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = SharedMLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = SharedMLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch):
        edge_index = knn_graph(pos, self.num_neighbors, batch=batch, loop=True)

        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out//8
        x = self.lfa1(edge_index, x, pos)  # N, d_out//2
        x = self.lfa2(edge_index, x, pos)  # N, d_out//2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out

        return x, pos, batch


def decimation_indices(ptr: LongTensor, decimation_factor: Number) -> Tuple[Tensor, LongTensor]:
    """Get indices which downsample each point cloud by a decimation factor.

    Decimation happens separately for each cloud to prevent emptying smaller
    point clouds. Empty clouds are prevented: clouds will have a least
    one node after decimation.

    Args:
        ptr (LongTensor): indices of samples in the batch.
        decimation_factor (Number): value to divide number of nodes with.
            Should be higher than 1 for downsampling.

    :rtype: (:class:`Tensor`, :class:`LongTensor`): indices for downsampling
        and resulting updated ptr.

    """
    if decimation_factor < 1:
        raise ValueError(
            "Argument `decimation_factor` should be higher than (or equal to) "
            f"1 for downsampling. (Current value: {decimation_factor})"
        )

    batch_size = ptr.size(0) - 1
    bincount = ptr[1:] - ptr[:-1]
    decimated_bincount = torch.div(bincount, decimation_factor, rounding_mode="floor")
    # Decimation should not empty clouds completely.
    decimated_bincount = torch.max(torch.ones_like(decimated_bincount), decimated_bincount)
    idx_decim = torch.cat(
        [
            (ptr[i] + torch.randperm(bincount[i], device=ptr.device)[: decimated_bincount[i]])
            for i in range(batch_size)
        ],
        dim=0,
    )
    # Get updated ptr (e.g. for future decimations)
    ptr_decim = ptr.clone()
    for i in range(batch_size):
        ptr_decim[i + 1] = ptr_decim[i] + decimated_bincount[i]

    return idx_decim, ptr_decim


def decimate(tensors, ptr: Tensor, decimation_factor: int):
    """Decimate each element of the given tuple of tensors."""
    idx_decim, ptr_decim = decimation_indices(ptr, decimation_factor)
    tensors_decim = tuple(tensor[idx_decim] for tensor in tensors)
    return tensors_decim, ptr_decim


class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


def main():
    category = "Airplane"  # Pass in `None` to train on all categories.
    category_num_classes = 4  # 4 for Airplane - see ShapeNet for details
    path = osp.join(
        osp.dirname(osp.realpath(__file__)),
        "..",
        "..",
        "..",
        "data",
        "ShapeNet",
    )
    transform = T.Compose(
        [
            T.RandomJitter(0.01),
            T.RandomRotate(15, axis=0),
            T.RandomRotate(15, axis=1),
            T.RandomRotate(15, axis=2),
        ]
    )
    pre_transform = T.NormalizeScale()
    train_dataset = ShapeNet(
        path,
        category,
        split="trainval",
        transform=transform,
        pre_transform=pre_transform,
    )
    test_dataset = ShapeNet(path, category, split="test", pre_transform=pre_transform)
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyGRandLANet(3, category_num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def train():
        model.train()

        total_loss = correct_nodes = total_nodes = 0
        for i, data in tqdm(enumerate(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.pos, data.batch, data.ptr)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
            total_nodes += data.num_nodes

            if (i + 1) % 10 == 0:
                print(
                    f"[{i+1}/{len(train_loader)}] Loss: {total_loss / 10:.4f} "
                    f"Train Acc: {correct_nodes / total_nodes:.4f}"
                )
                total_loss = correct_nodes = total_nodes = 0

    @torch.no_grad()
    def test(loader):
        model.eval()

        ious, categories = [], []
        y_map = torch.empty(loader.dataset.num_classes, device=device).long()
        for data in loader:
            data = data.to(device)
            outs = model(data.x, data.pos, data.batch, data.ptr)

            sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()
            for out, y, category in zip(
                outs.split(sizes), data.y.split(sizes), data.category.tolist()
            ):
                category = list(ShapeNet.seg_classes.keys())[category]
                part = ShapeNet.seg_classes[category]
                part = torch.tensor(part, device=device)

                y_map[part] = torch.arange(part.size(0), device=device)

                iou = jaccard_index(
                    out[:, part].argmax(dim=-1),
                    y_map[y],
                    num_classes=part.size(0),
                    absent_score=1.0,
                )
                ious.append(iou)

            categories.append(data.category)

        iou = torch.tensor(ious, device=device)
        category = torch.cat(categories, dim=0)

        mean_iou = scatter(iou, category, reduce="mean")  # Per-category IoU.
        return float(mean_iou.mean())  # Global IoU.

    for epoch in range(1, 31):
        train()
        iou = test(test_loader)
        print(f"Epoch: {epoch:02d}, Test IoU: {iou:.4f}")


if __name__ == "__main__":
    main()
