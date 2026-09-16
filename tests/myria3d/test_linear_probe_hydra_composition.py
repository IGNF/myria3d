"""End-to-end Hydra composition + instantiation sanity check for the linear-probing
experiment configs -- complements the GridProbeModel/callbacks unit tests (which
construct those classes directly) by exercising the actual `experiment=linear_probe/*`
config wiring the user runs on Jean Zay (`hydra.utils.instantiate` on the composed
`model`/`callbacks`/`datamodule` groups), without needing real downstream data.
"""

import hydra
import pytest
import torch

from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask
from tests.conftest import make_default_hydra_cfg

EXPERIMENTS = [
    "dales_grid",
    "dales_seeds",
    "h3d_grid",
    "h3d_seeds",
    "eclair_grid",
    "eclair_seeds",
]


def _write_fake_backbone_checkpoint(tmp_path):
    backbone = PyGRandLANetMultiTask(
        num_features=5,
        task_configs={"segment": {"task_type": "semantic", "num_classes": 16}},
        decimation=4,
        num_neighbors=16,
    )
    state_dict = {f"model.{k}": v for k, v in backbone.state_dict().items()}
    ckpt_path = tmp_path / "fake_backbone.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return str(ckpt_path)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_linear_probe_experiment_composes_and_instantiates(experiment, tmp_path, monkeypatch):
    ckpt_path = _write_fake_backbone_checkpoint(tmp_path)
    monkeypatch.setenv("FLAIR3D_CKPT_PATH", ckpt_path)
    monkeypatch.setenv("DOWNSTREAM_DATA_ROOT", str(tmp_path / "downstream"))

    cfg = make_default_hydra_cfg(overrides=[f"experiment=linear_probe/{experiment}"])

    model = hydra.utils.instantiate(cfg.model)
    assert model is not None
    assert all(not p.requires_grad for p in model.backbone.parameters())

    instantiated_callbacks = 0
    for cb_conf in cfg.callbacks.values():
        if "_target_" not in cb_conf:
            continue
        callback = hydra.utils.instantiate(cb_conf)
        assert callback is not None
        instantiated_callbacks += 1
    assert (
        instantiated_callbacks >= 3
    )  # model_checkpoint, grid_probe_metrics, + winner/seed-tester

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    assert datamodule is not None
