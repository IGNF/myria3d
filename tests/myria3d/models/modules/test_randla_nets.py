import pytest
import torch
from torch_geometric.data import Batch, Data
from myria3d.models.modules.randla_net import RandLANet
from myria3d.models.modules.pyg_randla_net import PyGRandLANet


@pytest.mark.parametrize("num_graphs", [1, 2])
def test_fake_run_randlanet(num_graphs):
    """Documents expected data format and make a forward pass with RandLa-Net

    Model pass with "batch_size=1" is a edge case that needs to pass to avoid unexpected crash due
    to incomplete batch at the end of an inference.

    """
    num_euclidian_dimensions = 3
    num_features = 9
    num_classes = 6
    hparams_net = {
        "d_in": num_features,
        "num_neighbors": 16,
        "decimation": 4,
        "dropout": 0.0,
        "num_classes": num_classes,
    }
    data = Batch()
    num_points = 12500
    data.batch = torch.concat(
        [torch.full((num_points,), i) for i in range(num_graphs)]
    )
    data.pos = torch.rand(
        (num_points * data.num_graphs, num_euclidian_dimensions)
    )
    data.x = torch.rand((num_points * data.num_graphs, num_features))
    model = RandLANet(**hparams_net)
    output = model(data.x, data.pos, data.batch, None)
    assert output.shape == torch.Size(
        [num_points * data.num_graphs, num_classes]
    )


@pytest.mark.parametrize(
    "num_nodes", [[12500, 12500], [50, 50], [12500, 10000]]
)
def test_fake_run_pyg_randlanet(num_nodes):
    """Documents expected data format and make a forward pass with PyG RandLa-Net.

    Accepts small clouds even though decimation should lead to empty cloud.
    Accepts point clouds of various sizes.

    """
    num_euclidian_dimensions = 3
    num_features = 9
    num_classes = 6
    decimation = 4
    num_neighbors = 16

    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand((n, num_features)),
                pos=torch.rand((n, num_euclidian_dimensions)),
                batch=torch.full((n,), idx),
            )
            for idx, n in enumerate(num_nodes)
        ]
    )

    model = PyGRandLANet(
        num_features,
        num_classes,
        decimation=decimation,
        num_neighbors=num_neighbors,
    )
    output = model(data.x, data.pos, data.batch, data.ptr)
    assert output.shape == torch.Size([sum(num_nodes), num_classes])
