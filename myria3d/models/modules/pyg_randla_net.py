from typing import List, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LeakyReLU, Sequential, Linear
from tqdm import tqdm

import torch_geometric.transforms as T
from torchmetrics.functional import jaccard_index
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool, knn_interpolate, BatchNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn
from torch_geometric.datasets import ShapeNet

# Default activation, BatchNorm, and resulting MLP used by RandLaNet authors
lrelu02_kwargs = {"negative_slope": 0.2}


bn099_kwargs = {"momentum": 0.99, "eps": 1e-6}


def default_MLP(*args, **kwargs):
    """MLP with custom activation, bn, and dropout that are always active even an last layer."""
    kwargs["plain_last"] = kwargs.get("plain_last", False)
    kwargs["act"] = kwargs.get("act", "LeakyReLU")
    kwargs["act_kwargs"] = kwargs.get("act_kwargs", lrelu02_kwargs)
    kwargs["norm_kwargs"] = kwargs.get("norm_kwargs", bn099_kwargs)
    return MLP(*args, **kwargs)


class PyGRandLANet(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        decimation: int = 4,
        num_neighbors: int = 16,
        return_logits: bool = False,
    ):
        super().__init__()

        self.return_logits = return_logits
        # Authors use 8, which might become a bottlenecj if num_classes>8 or num_features>8.
        bottleneck = max(num_classes, num_features)
        d = decimation
        nk = num_neighbors

        self.fc0 = Linear(num_features, bottleneck)
        self.lfa1_module = DilatedResidualBlock(d, nk, bottleneck, 32)
        self.lfa2_module = DilatedResidualBlock(d, nk, 32, 128)
        self.lfa3_module = DilatedResidualBlock(d, nk, 128, 256)
        self.lfa4_module = DilatedResidualBlock(d, nk, 256, 512)
        self.mlp1 = default_MLP([512, 512])
        self.fp4_module = FPModule(1, default_MLP([512 + 256, 256]))
        self.fp3_module = FPModule(1, default_MLP([256 + 128, 128]))
        self.fp2_module = FPModule(1, default_MLP([128 + 32, 32]))
        self.fp1_module = FPModule(1, default_MLP([32 + bottleneck, bottleneck]))

        self.mlp2 = Sequential(
            default_MLP([bottleneck, 64, 32], dropout=[0.0, 0.5]),
            Linear(32, num_classes),
        )

    def forward(self, batch):

        in_0 = (self.fc0(batch.x), batch.pos, batch.batch)

        lfa1_out = self.lfa1_module(*in_0)
        lfa2_out = self.lfa2_module(*lfa1_out)
        lfa3_out = self.lfa3_module(*lfa2_out)
        lfa4_out = self.lfa4_module(*lfa3_out)

        mlp_out = (self.mlp1(lfa4_out[0]), lfa4_out[1], lfa4_out[2])

        fp4_out = self.fp4_module(*mlp_out, *lfa3_out)
        fp3_out = self.fp3_module(*fp4_out, *lfa2_out)
        fp2_out = self.fp2_module(*fp3_out, *lfa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *in_0)

        logits = self.mlp2(x)
        if self.return_logits:
            return logits
        return logits.log_softmax(dim=-1)


# Default activation and BatchNorm used by RandLaNet authors
lrelu02 = LeakyReLU(negative_slope=0.2)


def bn099(in_channels):
    return BatchNorm(in_channels, momentum=0.99, eps=1e-6)


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, d_out):
        super().__init__(aggr="add", flow="target_to_source")
        self.mlp_encoder = default_MLP([10, d_out // 2])
        self.mlp_attention = default_MLP(
            [d_out, d_out], bias=False, act=None, norm=None
        )
        self.mlp_post_attention = default_MLP([d_out, d_out])

    def forward(self, edge_indx, x, pos):
        out = self.propagate(edge_indx, x=x, pos=pos)  # N // 4 * d_out
        out = self.mlp_post_attention(out)  # N // 4 * d_out
        return out

    def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
        """
        Local Spatial Encoding (locSE) and attentive pooling of features.

        Args:
            x_j (Tensor): neighboors features (K,d)
            pos_i (Tensor): centroid position (repeated) (K,3)
            pos_j (Tensor): neighboors positions (K,3)

        returns:
            (Tensor): locSE weighted by feature attention scores.

        """
        dist = pos_j - pos_i
        euclidian_dist = torch.sqrt((dist * dist).sum(1, keepdim=True))
        relative_infos = torch.cat(
            [pos_i, pos_j, dist, euclidian_dist], dim=1
        )  # N//4 * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N//4 * K, d
        local_features = torch.cat([x_j, local_spatial_encoding], dim=1)  # N//4 * K, 2d

        # attention will weight the different features of x
        attention_scores = torch.softmax(
            self.mlp_attention(local_features), dim=-1
        )  # N//4 * K, d_out
        return attention_scores * local_features  # N//4 * K, d_out


class DilatedResidualBlock(MessagePassing):
    def __init__(
        self,
        decimation,
        num_neighbors,
        d_in: int,
        d_out: int,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = default_MLP([d_in, d_out // 8])  # TODO: remove the norm ?
        # MLP on input, and the result is summed with the output of mlp2
        self.shortcut = default_MLP([d_in, d_out], act=None)
        # MLP on output
        self.mlp2 = default_MLP([d_out // 2, d_out], act=None)

        self.lfa1 = LocalFeatureAggregation(d_out // 4)
        self.lfa2 = LocalFeatureAggregation(d_out // 2)

        self.lrelu = torch.nn.LeakyReLU(**lrelu02_kwargs)

    def forward(self, x, pos, batch):
        # Random Sampling by decimation
        idx = subsample_by_decimation(batch, self.decimation)
        row, col = knn(
            pos, pos[idx], self.num_neighbors, batch_x=batch, batch_y=batch[idx]
        )
        edge_index = torch.stack([col, row], dim=0)
        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out // 8
        x = self.lfa1(edge_index, x, pos)  # N, d_out // 2
        x = self.lfa2(edge_index, x, pos)  # N, d_out // 2
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out
        return x[idx], pos[idx], batch[idx]  # N // decimation, d_out


def subsample_by_decimation(batch, decimation):
    """Subsamples by a decimation factor.

    Each sample needs to be decimated separately to prevent emptying point clouds by accident.

    """
    ends = (
        (torch.argwhere(torch.diff(batch) != 0) + 1)
        .cpu()
        .numpy()
        .squeeze()
        .astype(int)
        .tolist()
    )
    starts = [0] + ends
    ends = ends + [batch.size(0)]
    idx = torch.cat(
        [
            (start + torch.randperm(end - start))[::decimation]
            for start, end in zip(starts, ends)
        ],
        dim=0,
    )
    return idx


class FPModule(torch.nn.Module):
    """Upsampling with a skip connection."""

    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


def main():
    """This is for testing architecture on a dataset, even though it is not really adapated and quickly plateaus."""

    category = "Airplane"  # Pass in `None` to train on all categories.
    # path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "data", "ShapeNet")
    path = "/var/data/cgaydon/data/shapenet"
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
    model = PyGRandLANet(3, train_dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train():
        model.train()

        total_loss = correct_nodes = total_nodes = 0
        for i, data in tqdm(enumerate(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
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
            outs = model(data)

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
