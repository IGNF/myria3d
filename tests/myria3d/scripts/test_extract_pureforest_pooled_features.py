import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import extract_pureforest_pooled_features as extract  # noqa: E402

from myria3d.models.modules.hypercolumn import STAGE_CHANNELS  # noqa: E402
from myria3d.models.modules.pyg_randla_net_multitask import (  # noqa: E402
    PyGRandLANetMultiTask,
)
from myria3d.pctl.dataset.downstream.pureforest import (  # noqa: E402
    COORD_SCALE_M,
    load_flair3d_leakage_excluded_tiles,
    load_pureforest_scene,
)

NUM_FEATURES = 5
PRETRAIN_TASK_CONFIGS = {"segment": {"task_type": "semantic", "num_classes": 16}}
FEAT_DIM = sum(STAGE_CHANNELS.values())


def _write_checkpoint(tmp_path):
    backbone = PyGRandLANetMultiTask(
        NUM_FEATURES, PRETRAIN_TASK_CONFIGS, decimation=4, num_neighbors=4
    )
    state_dict = {f"model.{k}": v for k, v in backbone.state_dict().items()}
    state_dict["strength_mask_value"] = torch.tensor([[0.42]])
    ckpt_path = tmp_path / "backbone.ckpt"
    torch.save({"state_dict": state_dict}, ckpt_path)
    return str(ckpt_path)


def _write_pureforest_tile(data_root: Path, split: str, name: str, category: int, n: int = 200):
    scene_dir = data_root / split / name
    scene_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    # On-disk convention: mean-centered XY, min-shifted Z, /COORD_SCALE_M -- stays well
    # within [-1, 1] for a synthetic tile, same as real PureForest tiles.
    coord = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    color = rng.uniform(0.0, 255.0, size=(n, 3)).astype(np.float32)
    np.save(scene_dir / "coord.npy", coord)
    np.save(scene_dir / "color.npy", color)
    np.save(scene_dir / "category.npy", np.array(category, dtype=np.int32))
    return name


def test_extraction_writes_npz_with_expected_schema(tmp_path):
    ckpt_path = _write_checkpoint(tmp_path)
    data_root = tmp_path / "pureforest"
    for i in range(3):
        _write_pureforest_tile(data_root, "test", f"Fagus_sylvatica-C2-{i}", category=2)

    output_dir = tmp_path / "out"
    backbone, mask_values = extract.load_frozen_backbone(
        ckpt_path, num_features=NUM_FEATURES, num_neighbors=4, decimation=4
    )
    dataset = extract.PureForestDataset(str(data_root), "test")
    transform = extract.build_extract_transform(voxel=0.1)

    payload = extract.run_split(
        backbone=backbone,
        mask_values=mask_values,
        dataset=dataset,
        transform=transform,
        batch_size=2,
        device=torch.device("cpu"),
    )
    extract.save_split_npz(
        output_dir / "test.npz",
        payload,
        {
            "split": "test",
            "feat_dim": extract.FEAT_DIM,
            "channel_blocks": list(extract.CHANNEL_BLOCKS),
        },
    )

    npz = np.load(output_dir / "test.npz", allow_pickle=True)
    assert set(npz.files) >= {"names", "category", "mean_feat", "max_feat", "class_names"}
    assert npz["names"].shape[0] == 3
    assert npz["category"].dtype == np.int64
    assert npz["mean_feat"].dtype == np.float16
    assert npz["max_feat"].dtype == np.float16
    assert npz["mean_feat"].shape == (3, FEAT_DIM)
    assert npz["max_feat"].shape == (3, FEAT_DIM)
    assert sum(extract.CHANNEL_BLOCKS) == extract.FEAT_DIM == FEAT_DIM


def test_test_split_leakage_filter_drops_excluded_tiles(tmp_path):
    excluded_name = next(iter(load_flair3d_leakage_excluded_tiles()))
    data_root = tmp_path / "pureforest"
    _write_pureforest_tile(data_root, "test", excluded_name, category=0)
    _write_pureforest_tile(data_root, "test", "Fagus_sylvatica-C2-keepme", category=2)

    dataset = extract.PureForestDataset(str(data_root), "test", exclude_flair3d_leakage_tiles=True)

    assert excluded_name not in dataset.names
    assert "Fagus_sylvatica-C2-keepme" in dataset.names


def test_train_split_never_filtered_by_leakage_list(tmp_path):
    excluded_name = next(iter(load_flair3d_leakage_excluded_tiles()))
    data_root = tmp_path / "pureforest"
    _write_pureforest_tile(data_root, "train", excluded_name, category=0)

    dataset = extract.PureForestDataset(
        str(data_root), "train", exclude_flair3d_leakage_tiles=False
    )

    assert excluded_name in dataset.names


def test_coord_denormalized_by_coord_scale_m(tmp_path):
    data_root = tmp_path / "pureforest"
    _write_pureforest_tile(data_root, "train", "tile0", category=0, n=10)
    raw = np.load(data_root / "train" / "tile0" / "coord.npy")

    data = load_pureforest_scene(str(data_root / "train" / "tile0"))

    assert torch.allclose(data.pos, torch.from_numpy(raw * COORD_SCALE_M))
