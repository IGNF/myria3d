import pytest
import torch
from torch_geometric.data import Batch, Data

from myria3d.models.modules.pyg_randla_net import PyGRandLANet



# Check that using the new version of the code returns
# the same model structure with the default parameters as the older version
def test_pyg_randlanet_model_structure():
    num_features = 9
    num_classes = 6
    decimation = 4
    num_neighbors = 16

    model1 = PyGRandLANet(
        num_features,
        num_classes,
        decimation=decimation,
        num_neighbors=num_neighbors,
    )

    model2 = PyGRandLANet(
        num_features,
        num_classes,
        decimation=decimation,
        num_neighbors=num_neighbors,
        dims=[32, 128, 256, 512],
    )

    # The default dims are (32, 128, 256, 512), so passing them explicitly must
    # yield the same per-block input/output channel dimensions.
    for block_name in ("block1", "block2", "block3", "block4"):
        block1 = getattr(model1, block_name)
        block2 = getattr(model2, block_name)
        assert block1.d_in == block2.d_in
        assert block1.d_out == block2.d_out

    # The first linear layer (fc0) maps features to the bottleneck dimension.
    assert model1.fc0.in_features == model2.fc0.in_features
    assert model1.fc0.out_features == model2.fc0.out_features

    # Explicitly check the expected default embedding dimensions.
    assert model1.block1.d_out == 32
    assert model1.block2.d_out == 128
    assert model1.block3.d_out == 256
    assert model1.block4.d_out == 512

    # model2
    assert model2.block1.d_out == 32
    assert model2.block2.d_out == 128
    assert model2.block3.d_out == 256
    assert model2.block4.d_out == 512


@pytest.mark.parametrize("num_nodes", [[12500, 12500], [50, 50], [12500, 10000]])
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


# test model with externalized embeddings sizes defined as a list of integers as dims = [48, 192, 384, 768]
@pytest.mark.parametrize("num_nodes", [[12500, 12500], [50, 50], [12500, 10000]])
def test_fake_run_pyg_randlanet_with_externalized_embeddings(num_nodes):
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
        dims=[48,192,384,768],
    )
    output = model(data.x, data.pos, data.batch, data.ptr)
    assert output.shape == torch.Size([sum(num_nodes), num_classes])
