import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LeakyReLU, Sequential
from tqdm import tqdm

import torch_geometric.transforms as T
from torchmetrics.functional import jaccard_index
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, global_max_pool, knn_interpolate
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import knn
from torch_geometric.datasets import ShapeNet

from myria3d.models.modules.randla_net import get_batch_tensor_by_enumeration


class PyGRandLANet(torch.nn.Module):
    # num_features = total including pos, which should be removed.. TODO: make this simpler.
    def __init__(
        self,
        num_features,
        num_classes,
        decimation: int = 4,
        num_neighbors: int = 16,
        interpolation_k: int = 25,
        num_workers: int = 1,
    ):
        super().__init__()
        self.interpolation_k = interpolation_k
        self.num_workers = num_workers

        self.lfa1_module = DilatedResidualBlock(
            decimation, num_neighbors, num_features - 3, 32
        )
        self.lfa2_module = DilatedResidualBlock(decimation, num_neighbors, 32, 128)
        self.lfa3_module = DilatedResidualBlock(decimation, num_neighbors, 128, 256)
        self.lfa4_module = DilatedResidualBlock(decimation, num_neighbors, 256, 512)
        self.mlp1 = MLP([512, 512])
        self.fp4_module = FPModule(1, MLP([512 + 256, 256]))
        self.fp3_module = FPModule(1, MLP([256 + 128, 128]))
        self.fp2_module = FPModule(1, MLP([128 + 32, 32]))
        self.fp1_module = FPModule(1, MLP([32 + num_features - 3, 8]))

        self.mlp2 = MLP([8, 64, 32], dropout=0.5)
        self.lin = torch.nn.Linear(32, num_classes)

    def forward(self, batch):
        in_0 = (batch.x, batch.pos, batch.batch)

        lfa1_out = self.lfa1_module(*in_0)
        lfa2_out = self.lfa2_module(*lfa1_out)
        lfa3_out = self.lfa3_module(*lfa2_out)
        lfa4_out = self.lfa4_module(*lfa3_out)

        mlp_out = (self.mlp1(lfa4_out[0]), lfa4_out[1], lfa4_out[2])

        fp4_out = self.fp4_module(*mlp_out, *lfa3_out)
        fp3_out = self.fp3_module(*fp4_out, *lfa2_out)
        fp2_out = self.fp2_module(*fp3_out, *lfa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *in_0)

        x = self.mlp2(x)

        scores = self.lin(x)

        if self.training or "copies" not in batch:
            # In training mode and for validation, we directly optimize on subsampled points, for
            # 1) Speed of training - because interpolation multiplies a step duration by a 5-10 factor!
            # 2) data augmentation at the supervision level.
            return scores  # B*N, C

        # During evaluation on test data and inference, we interpolate predictions back to original positions
        # KNN is way faster on CPU than on GPU by a 3 to 4 factor.
        scores = scores.cpu()
        batch_y = get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)
        scores = knn_interpolate(
            scores.cpu(),
            batch.copies["pos_sampled_copy"].cpu(),
            batch.copies["pos_copy"].cpu(),
            batch_x=batch.batch.cpu(),
            batch_y=batch_y.cpu(),
            k=self.interpolation_k,
            num_workers=self.num_workers,
        )
        return scores  # N1+N2+...+Nn, C


class GlobalPooling(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(x)
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class LocalFeatureAggregation(MessagePassing):
    """Positional encoding of points in a neighborhood."""

    def __init__(self, d_out):
        # TODO: check if need for batch, activation, etc.
        super().__init__(aggr="add")
        self.mlp_encoder = MLP([10, d_out // 2])
        self.mlp_attention = MLP([d_out, d_out])

    def forward(self, edge_indx, x, pos):
        out = self.propagate(edge_indx, x=x, pos=pos)  # N // 4 * d_out
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
        euclidian_dist = torch.sqrt(dist * dist).sum(1, keepdim=True)
        relative_infos = torch.cat(
            [pos_i, pos_j, dist, euclidian_dist], dim=1
        )  # N//4 * K, d
        local_spatial_encoding = self.mlp_encoder(relative_infos)  # N//4 * K, d
        out1 = torch.cat([x_j, local_spatial_encoding], dim=1)  # N//4 * K, 2d

        # attention will weight the differetn features of x
        attention_scores = torch.softmax(
            self.mlp_attention(out1), dim=-1
        )  # N//4 * K, d_out
        out2 = attention_scores * out1  # N//4 * K, d_out
        return out2


class DilatedResidualBlock(MessagePassing):
    def __init__(self, decimation, num_neighbors, d_in, d_out):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.d_in = d_in
        self.d_out = d_out

        # MLP on input
        self.mlp1 = Sequential(
            MLP(
                [d_in, d_out // 4],
                norm=False,
            ),
            LeakyReLU(negative_slope=0.2),
        )
        # MLP on input whose result is summed with the output of mlp2
        self.shortcut = Sequential(
            MLP(
                [d_in, d_out],
            ),
            LeakyReLU(negative_slope=0.2),
        )
        # MLP on output
        self.mlp2 = Sequential(
            MLP(
                [d_out, d_out],
                norm=False,
            ),
            LeakyReLU(negative_slope=0.2),
        )

        self.lfa1 = LocalFeatureAggregation(d_out // 2)
        self.lfa2 = LocalFeatureAggregation(d_out)

        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x, pos, batch):
        # Random Sampling by decimation
        idx = torch.arange(start=0, end=batch.size(0), step=self.decimation)
        row, col = knn(
            pos, pos[idx], self.num_neighbors, batch_x=batch, batch_y=batch[idx]
        )
        edge_index = torch.stack([col, row], dim=0)
        shortcut_of_x = self.shortcut(x)  # N, d_out
        x = self.mlp1(x)  # N, d_out // 4
        x = self.lfa1(edge_index, x, pos)  # N, d_out
        x = self.lfa2(edge_index, x, pos)  # N, d_out
        x = self.mlp2(x)  # N, d_out
        x = self.lrelu(x + shortcut_of_x)  # N, d_out
        x, pos, batch = x[idx], pos[idx], batch[idx]  # N // decimation, d_out
        return x, pos, batch


class FPModule(torch.nn.Module):
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
