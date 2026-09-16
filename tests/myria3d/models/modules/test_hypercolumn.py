import torch
from torch_geometric.data import Batch, Data

from myria3d.models.modules.hypercolumn import DEFAULT_SCALES, hypercolumn_concat
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask


def _make_model_and_batch(num_features=5, n_per_graph=64):
    task_configs = {"segment": {"task_type": "semantic", "num_classes": 6}}
    model = PyGRandLANetMultiTask(num_features, task_configs, decimation=4, num_neighbors=4)
    data = Batch.from_data_list(
        [
            Data(
                x=torch.rand((n_per_graph, num_features)),
                pos=torch.rand((n_per_graph, 3)),
                batch=torch.full((n_per_graph,), idx),
            )
            for idx in range(2)
        ]
    )
    return model, data


def test_forward_encoder_stages_returns_block1_to_4_at_their_own_resolution():
    model, data = _make_model_and_batch()
    stages = model._forward_encoder_stages(data.x, data.pos, data.batch, data.ptr)

    assert set(stages.keys()) == {"enc1", "enc2", "enc3", "enc4"}
    expected_channels = {"enc1": 32, "enc2": 128, "enc3": 256, "enc4": 512}
    prev_num_nodes = data.num_nodes
    for name in ("enc1", "enc2", "enc3", "enc4"):
        feat, pos, batch = stages[name]
        assert feat.size(1) == expected_channels[name]
        assert feat.size(0) == pos.size(0) == batch.size(0)
        # Each stage is decimated relative to the previous one (block1 == full input).
        assert feat.size(0) <= prev_num_nodes
        prev_num_nodes = feat.size(0)


def test_hypercolumn_concat_channel_count_and_target_resolution():
    model, data = _make_model_and_batch()
    stages = model._forward_encoder_stages(data.x, data.pos, data.batch, data.ptr)

    feat = hypercolumn_concat(stages, data.pos, data.batch, scales=DEFAULT_SCALES, k=1)

    assert feat.shape == (data.num_nodes, 32 + 128 + 256 + 512)


def test_hypercolumn_concat_subset_of_scales():
    model, data = _make_model_and_batch()
    stages = model._forward_encoder_stages(data.x, data.pos, data.batch, data.ptr)

    feat = hypercolumn_concat(stages, data.pos, data.batch, scales=("enc1", "enc4"), k=1)

    assert feat.shape == (data.num_nodes, 32 + 512)
