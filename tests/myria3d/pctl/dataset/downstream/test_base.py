import os

import numpy as np
import pytest
import torch

from myria3d.pctl.dataset.downstream.base import (
    DownstreamNpyDataset,
    build_scene_entries,
    list_scene_dirs,
    load_downstream_scene,
)


def _write_scene(root, split, name, n=16, with_color=True, with_strength=True, with_label=True):
    scene_dir = os.path.join(root, split, name)
    os.makedirs(scene_dir, exist_ok=True)
    np.save(os.path.join(scene_dir, "coord.npy"), np.random.rand(n, 3).astype(np.float32))
    if with_color:
        np.save(
            os.path.join(scene_dir, "color.npy"),
            np.random.randint(0, 255, size=(n, 3)).astype(np.uint8),
        )
    if with_strength:
        np.save(
            os.path.join(scene_dir, "strength.npy"),
            np.random.rand(n).astype(np.float32),
        )
    if with_label:
        np.save(
            os.path.join(scene_dir, "segment.npy"),
            np.random.randint(0, 5, size=(n,)).astype(np.int64),
        )
    return scene_dir


def test_list_scene_dirs(tmp_path):
    _write_scene(str(tmp_path), "train", "scene_a")
    _write_scene(str(tmp_path), "train", "scene_b")
    dirs = list_scene_dirs(str(tmp_path), "train")
    assert len(dirs) == 2


def test_load_downstream_scene_no_color_sets_color_mask(tmp_path):
    scene_dir = _write_scene(str(tmp_path), "train", "s", with_color=False)
    data = load_downstream_scene(scene_dir, has_color=False, has_strength=True)
    assert torch.equal(data.color_mask, torch.ones(data.x.size(0), dtype=torch.bool))
    assert not hasattr(data, "strength_mask")
    assert torch.equal(data.x[:, 1:4], torch.zeros(data.x.size(0), 3))


def test_load_downstream_scene_no_strength_sets_strength_mask(tmp_path):
    scene_dir = _write_scene(str(tmp_path), "train", "s", with_strength=False)
    data = load_downstream_scene(scene_dir, has_color=True, has_strength=False)
    assert torch.equal(data.strength_mask, torch.ones(data.x.size(0), dtype=torch.bool))
    assert not hasattr(data, "color_mask")
    assert torch.equal(data.x[:, 0], torch.zeros(data.x.size(0)))


def test_load_downstream_scene_passes_segment_through_unchanged(tmp_path):
    scene_dir = os.path.join(str(tmp_path), "scene")
    os.makedirs(scene_dir)
    np.save(os.path.join(scene_dir, "coord.npy"), np.zeros((4, 3), dtype=np.float32))
    np.save(os.path.join(scene_dir, "segment.npy"), np.array([0, 7, 8, 3], dtype=np.int32))
    data = load_downstream_scene(scene_dir, has_color=False, has_strength=False)
    assert torch.equal(data.y, torch.tensor([0, 7, 8, 3]))


def test_build_scene_entries_train_is_one_entry_per_scene(tmp_path):
    _write_scene(str(tmp_path), "train", "a")
    _write_scene(str(tmp_path), "train", "b")
    entries = build_scene_entries(str(tmp_path), "train", is_eval=False)
    assert len(entries) == 2
    assert all(subtile_index is None for _, subtile_index in entries)


def test_build_scene_entries_eval_expands_into_subtiles(tmp_path):
    _write_scene(str(tmp_path), "test", "a")
    entries = build_scene_entries(
        str(tmp_path), "test", is_eval=True, tile_width=100, subtile_width=50
    )
    assert len(entries) == 4
    assert [subtile_index for _, subtile_index in entries] == [0, 1, 2, 3]


def test_downstream_npy_dataset_raises_on_empty_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        DownstreamNpyDataset(data_root=str(tmp_path), split_dir="train")


def test_downstream_npy_dataset_sets_patch_id(tmp_path):
    _write_scene(str(tmp_path), "train", "scene_a")
    dataset = DownstreamNpyDataset(data_root=str(tmp_path), split_dir="train")
    data = dataset[0]
    assert data.patch_id == "scene_a"
