import functools
import json

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from myria3d.callbacks.grid_probe_callbacks import (
    GridProbeMetrics,
    GridProbeSeedEnsembleTester,
    GridProbeWinnerSelector,
)
from myria3d.models.grid_probe_model import GridProbeModel
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask

NUM_FEATURES = 5
NUM_CLASSES = 4
PRETRAIN_TASK_CONFIGS = {"segment": {"task_type": "semantic", "num_classes": 16}}


def _make_fake_checkpoint(tmp_path):
    backbone = PyGRandLANetMultiTask(
        NUM_FEATURES, PRETRAIN_TASK_CONFIGS, decimation=4, num_neighbors=4
    )
    state_dict = {f"model.{k}": v for k, v in backbone.state_dict().items()}
    ckpt_path = tmp_path / "fake_backbone.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return str(ckpt_path)


class _PrebatchedDataset(Dataset):
    def __init__(self, batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


def _make_batch(n_per_graph=48, num_graphs=2):
    data_list = []
    for graph_idx in range(num_graphs):
        n = n_per_graph
        data_list.append(
            Data(
                x=torch.rand((n, NUM_FEATURES)),
                pos=torch.rand((n, 3)),
                y=torch.randint(0, NUM_CLASSES, (n,)),
                batch=torch.full((n,), graph_idx),
                x_features_names=["Intensity", "Red", "Green", "Blue", "rgb_avg"],
            )
        )
    return Batch.from_data_list(data_list)


def _make_model(ckpt_path, probe_lrs):
    return GridProbeModel(
        backbone_ckpt_path=ckpt_path,
        probe_lrs=list(probe_lrs),
        num_classes=NUM_CLASSES,
        ignore_index=None,
        num_features=NUM_FEATURES,
        num_neighbors=4,
        decimation=4,
        optimizer=functools.partial(torch.optim.Adam),
    )


def _make_loader(num_batches=3):
    return DataLoader(
        _PrebatchedDataset([_make_batch() for _ in range(num_batches)]), batch_size=None
    )


def test_grid_probe_winner_selector_writes_results_and_sets_active_probe(tmp_path):
    ckpt_path = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path, probe_lrs=[1e-2, 1e-1, 5e-1])
    output_path = tmp_path / "grid_search_results.json"

    metrics_cb = GridProbeMetrics(num_classes=NUM_CLASSES)
    winner_cb = GridProbeWinnerSelector(select_metric="mIoU", output_path=str(output_path))

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=2,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[metrics_cb, winner_cb],
    )
    trainer.fit(model, train_dataloaders=_make_loader(), val_dataloaders=_make_loader(2))

    assert output_path.is_file()
    results = json.loads(output_path.read_text())
    assert results["select_metric"] == "mIoU"
    assert results["winner"]["probe"] in model.probe_names
    assert len(results["leaderboard"]) == len(model.probe_names)
    assert model.active_probe == results["winner"]["probe"]


def test_grid_probe_seed_ensemble_tester_aggregates_across_probes(tmp_path):
    ckpt_path = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path, probe_lrs=[1e-1] * 4)  # 4 "seeds" at the same LR.
    results_path = tmp_path / "seed_ensemble_results.json"
    csv_path = tmp_path / "grid_then_seeds_summary.csv"

    metrics_cb = GridProbeMetrics(num_classes=NUM_CLASSES)
    seed_cb = GridProbeSeedEnsembleTester(
        results_path=str(results_path), summary_csv_path=str(csv_path), dataset_name="toy"
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[metrics_cb, seed_cb],
    )
    trainer.fit(model, train_dataloaders=_make_loader(1))
    trainer.test(model, dataloaders=_make_loader(2))

    assert results_path.is_file()
    results = json.loads(results_path.read_text())
    assert set(results["per_seed"].keys()) == set(model.probe_names)
    assert "mIoU" in results["aggregate"]
    for stat in ("mean", "std", "min", "max", "n"):
        assert stat in results["aggregate"]["mIoU"]
    assert results["aggregate"]["mIoU"]["n"] == len(model.probe_names)

    assert csv_path.is_file()
    with open(csv_path) as f:
        rows = list(f)
    assert len(rows) == 2  # header + one row
    assert "toy" in rows[1]


def test_grid_probe_metrics_logs_per_probe_tags(tmp_path):
    ckpt_path = _make_fake_checkpoint(tmp_path)
    model = _make_model(ckpt_path, probe_lrs=[1e-2, 1e-1])
    metrics_cb = GridProbeMetrics(num_classes=NUM_CLASSES)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[metrics_cb],
    )
    trainer.fit(model, train_dataloaders=_make_loader(1), val_dataloaders=_make_loader(1))

    for probe_name in model.probe_names:
        assert f"val/probe_{probe_name}/mIoU" in trainer.callback_metrics
        assert f"val/probe_{probe_name}/macro_f1" in trainer.callback_metrics
        assert f"val/probe_{probe_name}/acc" in trainer.callback_metrics
