import torch
from torch_geometric.data import Batch
from myria3d.models.modules.randla_net import RandLANet


def test_fake_run_randlanet():
    """Documents expected data format and make a forward pass with RandLa-Net"""
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
    batch.num_batches = 4
    batch_size = 12500
    batch.pos = torch.rand((batch_size * batch.num_batches, num_euclidian_dimensions))
    batch.x = torch.rand((batch_size * batch.num_batches, num_features))
    rln = RandLANet(hparams_net)
    output = rln(batch)
    assert output.shape == torch.Size([batch_size * batch.num_batches, num_classes])
