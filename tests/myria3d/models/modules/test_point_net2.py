import torch
from torch_geometric.data import Batch
from myria3d.models.modules.randla_net import RandLANet


def test_fake_run_pointnet2():
    """Documents expected data format and make a forward pass with RandLa-Net"""
    num_euclidian_dimensions = 3
    num_features = 9
    d_in = num_euclidian_dimensions + num_features
    num_classes = 6
    hparams_net = {
        "d_in": d_in,
        "r1": 2/50,
        "r2": 4/50,
        "num_classes": num_classes,
    }
    batch = Batch()
    batch.num_graphs = 4
    num_points = 12500
    batch.pos = torch.rand((num_points * batch.num_graphs, num_euclidian_dimensions))
    batch.x = torch.rand((num_points * batch.num_graphs, num_features))
    rln = RandLANet(hparams_net)
    output = rln(batch)
    assert output.shape == torch.Size([num_points * batch.num_graphs, num_classes])
