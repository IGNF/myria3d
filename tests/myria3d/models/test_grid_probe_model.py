import functools

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from myria3d.models.grid_probe_model import GridProbeModel, load_frozen_backbone
from myria3d.models.modules.hypercolumn import STAGE_CHANNELS
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask

NUM_FEATURES = 5
NUM_CLASSES = 4
PRETRAIN_TASK_CONFIGS = {
    "segment": {"task_type": "semantic", "num_classes": 16},
    "elevation": {"task_type": "regression"},
}


def _make_fake_checkpoint(tmp_path, learned_masked_feat=True):
    """A throwaway 'pretrained' backbone, saved the way MultiTaskModel would (weights
    prefixed 'model.', learned mask values as top-level LightningModule params)."""
    backbone = PyGRandLANetMultiTask(
        NUM_FEATURES, PRETRAIN_TASK_CONFIGS, decimation=4, num_neighbors=4
    )
    state_dict = {f"model.{k}": v for k, v in backbone.state_dict().items()}
    if learned_masked_feat:
        state_dict["color_mask_value"] = torch.tensor([[0.11, 0.22, 0.33, 0.44]])
        state_dict["strength_mask_value"] = torch.tensor([[0.77]])
    ckpt_path = tmp_path / "fake_backbone.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return str(ckpt_path), backbone


class _PrebatchedDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def _make_batch(n_per_graph=48, num_graphs=2, color_mask=None, strength_mask=None):
    data_list = []
    for graph_idx in range(num_graphs):
        n = n_per_graph
        kwargs = dict(
            x=torch.rand((n, NUM_FEATURES)),
            pos=torch.rand((n, 3)),
            y=torch.randint(0, NUM_CLASSES, (n,)),
            batch=torch.full((n,), graph_idx),
            x_features_names=["Intensity", "Red", "Green", "Blue", "rgb_avg"],
        )
        if color_mask is not None:
            kwargs["color_mask"] = torch.full((n,), color_mask, dtype=torch.bool)
        if strength_mask is not None:
            kwargs["strength_mask"] = torch.full((n,), strength_mask, dtype=torch.bool)
        data_list.append(Data(**kwargs))
    return Batch.from_data_list(data_list)


def _make_model(ckpt_path, probe_lrs=(1e-3, 1e-2, 1e-1), **overrides):
    kwargs = dict(
        backbone_ckpt_path=ckpt_path,
        probe_lrs=list(probe_lrs),
        num_classes=NUM_CLASSES,
        ignore_index=None,
        num_features=NUM_FEATURES,
        num_neighbors=4,
        decimation=4,
        optimizer=functools.partial(torch.optim.Adam),
    )
    kwargs.update(overrides)
    return GridProbeModel(**kwargs)


def test_load_frozen_backbone_freezes_params_and_loads_mask_values(tmp_path):
    ckpt_path, _ = _make_fake_checkpoint(tmp_path)
    backbone, mask_values = load_frozen_backbone(
        ckpt_path, num_features=NUM_FEATURES, num_neighbors=4
    )

    assert all(not p.requires_grad for p in backbone.parameters())
    assert not backbone.training
    assert set(mask_values.keys()) == {"color_mask_value", "strength_mask_value"}
    assert torch.equal(mask_values["color_mask_value"], torch.tensor([[0.11, 0.22, 0.33, 0.44]]))


def test_load_frozen_backbone_without_learned_mask(tmp_path):
    ckpt_path, _ = _make_fake_checkpoint(tmp_path, learned_masked_feat=False)
    _, mask_values = load_frozen_backbone(ckpt_path, num_features=NUM_FEATURES, num_neighbors=4)
    assert mask_values == {}


def test_hypercolumn_dim_matches_sum_of_scale_block_channels(tmp_path):
    ckpt_path, _ = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path, scale_blocks=("enc1", "enc3"))
    assert model.hypercolumn_dim == STAGE_CHANNELS["enc1"] + STAGE_CHANNELS["enc3"]
    for probe in model.probes.values():
        assert probe.net[-1].in_features == model.hypercolumn_dim


def test_masked_features_use_frozen_learned_fill_value(tmp_path):
    ckpt_path, _ = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path)

    # DALES-like: no color at all -> color_mask all True.
    batch = _make_batch(color_mask=True)
    x = model._fill_masked_features(batch)
    assert torch.allclose(x[:, 1], torch.full((x.size(0),), 0.11))
    assert torch.allclose(x[:, 2], torch.full((x.size(0),), 0.22))
    assert torch.allclose(x[:, 3], torch.full((x.size(0),), 0.33))

    # H3D-like: no intensity at all -> strength_mask all True.
    batch = _make_batch(strength_mask=True)
    x = model._fill_masked_features(batch)
    assert torch.allclose(x[:, 0], torch.full((x.size(0),), 0.77))


def test_training_step_freezes_backbone_and_isolates_probe_gradients(tmp_path):
    ckpt_path, _ = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path, probe_lrs=[1e-2, 1e-1, 5e-1])

    backbone_before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    probe_weights_before = {
        name: probe.net[-1].weight.detach().clone() for name, probe in model.probes.items()
    }

    batches = [_make_batch() for _ in range(3)]
    loader = DataLoader(_PrebatchedDataset(batches), batch_size=None)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=3,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=loader)

    # Backbone must be bit-identical: requires_grad=False truly enforced.
    for key, value in backbone_before.items():
        assert torch.equal(value, model.backbone.state_dict()[key])

    # Each probe's Linear weights must have moved, and diverged from one another
    # (gradient isolation -- the concrete check for the shared-backward correctness
    # risk: each head only sees its own gradient despite one summed backward()).
    after = {name: probe.net[-1].weight.detach().clone() for name, probe in model.probes.items()}
    for name in model.probe_names:
        assert not torch.equal(probe_weights_before[name], after[name])
    names = model.probe_names
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not torch.equal(after[names[i]], after[names[j]])


def test_active_probe_restricts_eval_to_one_probe(tmp_path):
    ckpt_path, _ = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path, probe_lrs=[1e-2, 1e-1])
    model.active_probe = model.probe_names[0]

    batch = _make_batch()
    out = model.validation_step(batch, 0)
    assert set(out["logits"].keys()) == {model.probe_names[0]}
