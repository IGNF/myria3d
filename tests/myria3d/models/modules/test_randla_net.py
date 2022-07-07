import pytest
import torch
from torch_geometric.data import Batch
from myria3d.models.modules.randla_net import RandLANet


@pytest.mark.parametrize("num_graphs", [1, 4])
def test_fake_run_randlanet(num_graphs):
    """Documents expected data format and make a forward pass with RandLa-Net

    Model pass with "batch_size=1" is a edge case that needs to pass to avoid unexpected crash due to incomplete batch at the
    end of an inference.

    """
    num_euclidian_dimensions = 3
    num_features = 9
    d_in = num_euclidian_dimensions + num_features
    num_classes = 6
    hparams_net = {
        "d_in": d_in,
        "num_neighbors": 16,
        "decimation": 4,
        "dropout": 0.0,
        "num_classes": num_classes,
    }
    batch = Batch()
    batch.num_graphs = 1
    num_points = 12500
    batch.pos = torch.rand((num_points * batch.num_graphs, num_euclidian_dimensions))
    batch.x = torch.rand((num_points * batch.num_graphs, num_features))
    rln = RandLANet(hparams_net)
    output = rln(batch)
    assert output.shape == torch.Size([num_points * batch.num_graphs, num_classes])
