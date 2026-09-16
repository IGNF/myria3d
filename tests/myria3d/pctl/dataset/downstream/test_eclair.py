import json
import os

import numpy as np
import pytest

from myria3d.pctl.dataset.downstream.eclair import EclairDataset


def _write_scene(root, split, name, n=8, review_category=None):
    scene_dir = os.path.join(root, split, name)
    os.makedirs(scene_dir, exist_ok=True)
    np.save(os.path.join(scene_dir, "coord.npy"), np.random.rand(n, 3).astype(np.float32))
    np.save(
        os.path.join(scene_dir, "color.npy"),
        np.random.randint(0, 255, size=(n, 3)).astype(np.uint8),
    )
    np.save(
        os.path.join(scene_dir, "strength.npy"),
        np.random.rand(n).astype(np.float32),
    )
    np.save(
        os.path.join(scene_dir, "segment.npy"),
        np.random.randint(0, 5, size=(n,)).astype(np.int64),
    )
    if review_category is not None:
        with open(os.path.join(scene_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"review_category": review_category}, f)
    return scene_dir


def test_include_pseudo_true_keeps_every_train_scene(tmp_path):
    _write_scene(str(tmp_path), "train", "approved_scene", review_category="approved")
    _write_scene(str(tmp_path), "train", "pseudo_scene", review_category="pseudo")
    _write_scene(str(tmp_path), "train", "no_meta_scene", review_category=None)

    dataset = EclairDataset(str(tmp_path), "train", include_pseudo=True)

    assert len(dataset) == 3


def test_include_pseudo_false_keeps_only_approved_scenes(tmp_path):
    _write_scene(str(tmp_path), "train", "approved_scene", review_category="approved")
    _write_scene(str(tmp_path), "train", "pseudo_scene", review_category="pseudo")
    _write_scene(str(tmp_path), "train", "rejected_scene", review_category="rejected")

    dataset = EclairDataset(str(tmp_path), "train", include_pseudo=False)

    assert len(dataset) == 1
    assert dataset[0].patch_id == "approved_scene"


def test_include_pseudo_false_raises_on_missing_meta_json(tmp_path):
    _write_scene(str(tmp_path), "train", "no_meta_scene", review_category=None)

    with pytest.raises(FileNotFoundError):
        EclairDataset(str(tmp_path), "train", include_pseudo=False)


def test_include_pseudo_filter_never_applied_outside_train_split(tmp_path):
    _write_scene(str(tmp_path), "val", "pseudo_scene", review_category="pseudo")

    # include_pseudo=False on a non-train split must not filter (and must not crash
    # on the absence of meta.json, since the filter is a no-op there).
    dataset = EclairDataset(str(tmp_path), "val", include_pseudo=False)

    assert len(dataset) == 1
