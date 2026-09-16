"""PureForest per-tile species-classification dataset, used only by
``scripts/extract_pureforest_pooled_features.py`` for pooled hypercolumn-feature
extraction -- PureForest's ``category`` label is scene-level, not per-point, so this
is a dedicated, lightweight loader (not a `DownstreamNpyDataset` subclass, which
assumes a per-point `y` and mosaic subtiling).
"""

from __future__ import annotations

import os
import os.path as osp
from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Pointcept's PureForest preprocessing normalization convention (mean-center XY,
# min-shift Z, divide all 3 axes by this) -- see
# pointcept/datasets/preprocessing/pureforest/preprocess_pureforest.py::normalize_tile_coord
# in the sibling Pointcept repo. Denormalized back to real meters on load (below).
COORD_SCALE_M = 25.0

COLOR_NORMALIZATION_MAX_VALUE = 255.0
X_FEATURE_NAMES = ["Intensity", "Red", "Green", "Blue", "rgb_avg"]

# 13 PureForest species, index-aligned with `category.npy` -- see
# pointcept/datasets/preprocessing/pureforest/pureforest_classes.py::CLASS_NAMES.
CLASS_NAMES = [
    "deciduous_oak",
    "evergreen_oak",
    "beech",
    "chestnut",
    "black_locust",
    "maritime_pine",
    "scotch_pine",
    "black_pine",
    "aleppo_pine",
    "fir",
    "spruce",
    "larch",
    "douglas",
]

# Copied verbatim from the sibling Pointcept repo (self-contained, no cross-repo
# runtime dependency) -- see the file for provenance.
_LEAKAGE_EXCLUDE_FILE = osp.join(
    osp.dirname(__file__), "pureforest_assets", "flair3d_leakage_excluded_test_tiles.txt"
)


@lru_cache(maxsize=1)
def load_flair3d_leakage_excluded_tiles() -> frozenset:
    """Tile (folder) names to drop from the PureForest *test* split only: their
    forest polygon (bdforetv2_id) geographically overlaps a Flair3D+ train/val tile,
    which would leak into the LP test set via the very backbone being probed."""
    with open(_LEAKAGE_EXCLUDE_FILE, encoding="utf-8") as f:
        return frozenset(line.strip() for line in f if line.strip() and not line.startswith("#"))


def list_pureforest_scene_names(data_root: str, split: str) -> List[str]:
    split_dir = osp.join(data_root, split)
    if not osp.isdir(split_dir):
        raise FileNotFoundError(f"PureForest split directory not found: {split_dir}")
    names = sorted(
        entry
        for entry in os.listdir(split_dir)
        if osp.isdir(osp.join(split_dir, entry))
        and osp.isfile(osp.join(split_dir, entry, "coord.npy"))
    )
    if not names:
        raise FileNotFoundError(f"No preprocessed PureForest scenes under {split_dir}")
    return names


def load_pureforest_scene(scene_dir: str) -> Data:
    """Load one PureForest scene folder into a PyG `Data` (no `y`: `category` is a
    single scene-level int, not a per-point label).

    On-disk `coord.npy` is Pointcept's own ``[-1, 1]``-ish pre-normalized convention
    -- denormalized back to real meters here (``* COORD_SCALE_M``) so myria3d's own
    point-budget / normalization transforms (which expect meter-space, e.g.
    `NormalizePos`'s `subtile_width` scaling) run unchanged, exactly like Pointcept's
    own `PureForestDataset.get_data` does.
    """
    coord = np.load(osp.join(scene_dir, "coord.npy")).astype(np.float32, copy=False)
    coord = coord * COORD_SCALE_M
    num_points = coord.shape[0]

    color_path = osp.join(scene_dir, "color.npy")
    color = np.load(color_path) if osp.isfile(color_path) else None
    if color is not None and color.shape[0] == num_points:
        red = color[:, 0].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        green = color[:, 1].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        blue = color[:, 2].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        rgb_avg = (red + green + blue) / 3.0
    else:
        red = green = blue = rgb_avg = np.zeros(num_points, dtype=np.float32)
    # PureForest's preprocessing writes no strength/intensity asset at all.
    intensity = np.zeros(num_points, dtype=np.float32)
    x = np.stack([intensity, red, green, blue, rgb_avg], axis=1).astype(np.float32, copy=False)

    category = int(np.load(osp.join(scene_dir, "category.npy")).reshape(-1)[0])

    return Data(
        pos=torch.from_numpy(coord),
        x=torch.from_numpy(x),
        x_features_names=list(X_FEATURE_NAMES),
        strength_mask=torch.ones(num_points, dtype=torch.bool),
        category=category,
    )


class PureForestDataset(Dataset):
    """Iterates PureForest scene folders for pooled-feature extraction."""

    def __init__(
        self,
        data_root: str,
        split: str,
        exclude_flair3d_leakage_tiles: bool = False,
    ):
        self.data_root = data_root
        self.split = split
        names = list_pureforest_scene_names(data_root, split)
        if exclude_flair3d_leakage_tiles:
            excluded = load_flair3d_leakage_excluded_tiles()
            filtered = [name for name in names if name not in excluded]
            if len(filtered) != len(names):
                from myria3d.utils import utils

                utils.get_logger(__name__).info(
                    "PureForest %s: excluded %d / %d Flair3D+-trainval-leaking tile(s).",
                    split,
                    len(names) - len(filtered),
                    len(names),
                )
            names = filtered
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Data:
        name = self.names[idx]
        data = load_pureforest_scene(osp.join(self.data_root, self.split, name))
        data.patch_id = name
        return data
