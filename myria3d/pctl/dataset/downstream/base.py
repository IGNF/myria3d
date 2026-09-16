"""Generic loader for Pointcept-preprocessed downstream linear-probing datasets
(DALES / H3D / ECLAIR): `{data_root}/{split_dir}/*/` folders containing `coord.npy`
plus whatever subset of `color.npy` / `strength.npy` / `segment.npy` exists.

Deliberately not built on `myria3d.pctl.dataset.pointcept_npy` -- that loader's
`build_scene_list` / `load_pointcept_scene` are tightly coupled to the Flair3D+ CSV
manifest schema and multitask target files (rasters, natural-habitat axes, ...) that
none of these single-task downstream datasets have.
"""

from __future__ import annotations

import glob
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from myria3d.pctl.dataset.utils import get_num_subtiles, pre_filter_below_n_points

# (scene_dir, subtile_index): subtile_index is None for train (SubtileCrop draws a
# random quadrant per epoch) and an explicit 0..N-1 for val/test (deterministic,
# full-tile coverage across N entries -- mirrors pointcept_npy.py's build_scene_list).
SceneEntry = Tuple[str, Optional[int]]

# Layout matches myria3d's classic RandLA-Net convention (and pointcept_npy.py):
# Intensity, Red, Green, Blue, rgb_avg -- the 5-feature schema the Flair3D+ backbone
# (d_in=5, configs/dataset_description/flair3d_plus_multitask.yaml) was pretrained on.
X_FEATURE_NAMES = ["Intensity", "Red", "Green", "Blue", "rgb_avg"]
COLOR_NORMALIZATION_MAX_VALUE = 255.0


def list_scene_dirs(data_root: str, split_dir: str) -> List[str]:
    """List Pointcept-preprocessed scene folders under ``{data_root}/{split_dir}/*/``."""
    pattern = osp.join(data_root, split_dir, "*")
    return sorted(d for d in glob.glob(pattern) if osp.isdir(d))


def build_scene_entries(
    data_root: str,
    split_dir: str,
    *,
    is_eval: bool,
    tile_width: Number = 50,
    subtile_width: Number = 50,
    subtile_overlap: Number = 0,
    scene_dirs: Optional[Sequence[str]] = None,
) -> List[SceneEntry]:
    """Build ``(scene_dir, subtile_index)`` entries for one split.

    Train yields one entry per scene (a random quadrant is drawn by `SubtileCrop` at
    transform time, a fresh draw every epoch). Val/test yield one entry per mosaic
    subtile (``subtile_index`` 0..N-1) for deterministic, full-tile coverage -- same
    convention as `pointcept_npy.py::build_scene_list`, without the CSV-manifest /
    exclusion-list machinery that only applies to the Flair3D+ pretraining pipeline.
    When ``tile_width == subtile_width`` there is exactly one subtile (the whole
    tile), so this is a no-op for datasets whose native tiles are already
    subtile-sized.
    """
    dirs = list(scene_dirs) if scene_dirs is not None else list_scene_dirs(data_root, split_dir)
    if not is_eval:
        return [(scene_dir, None) for scene_dir in dirs]

    num_subtiles = get_num_subtiles(tile_width, subtile_width, subtile_overlap=subtile_overlap)
    return [
        (scene_dir, subtile_index) for scene_dir in dirs for subtile_index in range(num_subtiles)
    ]


def _build_feature_matrix(
    color: Optional[np.ndarray],
    strength: Optional[np.ndarray],
    num_points: int,
    has_color: bool,
    has_strength: bool,
) -> np.ndarray:
    if has_strength and strength is not None:
        intensity = strength.reshape(-1).astype(np.float32, copy=False)
    else:
        intensity = np.zeros(num_points, dtype=np.float32)

    if has_color and color is not None and color.shape[0] == num_points:
        red = color[:, 0].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        green = color[:, 1].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        blue = color[:, 2].astype(np.float32, copy=False) / COLOR_NORMALIZATION_MAX_VALUE
        rgb_avg = (red + green + blue) / 3.0
    else:
        red = green = blue = rgb_avg = np.zeros(num_points, dtype=np.float32)

    return np.stack([intensity, red, green, blue, rgb_avg], axis=1).astype(np.float32, copy=False)


def load_downstream_scene(
    scene_dir: str,
    *,
    has_color: bool,
    has_strength: bool,
    label_key: str = "segment",
) -> Data:
    """Load one Pointcept-preprocessed downstream scene folder into a PyG `Data`.

    ``has_color``/``has_strength`` describe whether the *dataset as a whole* carries
    that modality (DALES has no color, H3D has no intensity) -- when a modality is
    absent, an all-True ``color_mask``/``strength_mask`` is attached so the frozen
    backbone's own learned fill-in value is used in place of a naive zero (see
    ``GridProbeModel._fill_masked_features``), matching how the backbone was
    pretrained (`RandomDropColor`/`RandomDropStrength`, see readme_flair3d.md).
    """
    coord_path = osp.join(scene_dir, "coord.npy")
    if not osp.isfile(coord_path):
        raise FileNotFoundError(f"Missing coord.npy in {scene_dir}")
    pos = np.load(coord_path).astype(np.float32, copy=False)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"coord.npy must be (N, 3), got {pos.shape} in {scene_dir}")
    num_points = pos.shape[0]

    color = None
    if has_color:
        color_path = osp.join(scene_dir, "color.npy")
        if osp.isfile(color_path):
            color = np.load(color_path)

    strength = None
    if has_strength:
        strength_path = osp.join(scene_dir, "strength.npy")
        if osp.isfile(strength_path):
            strength = np.load(strength_path)

    x = _build_feature_matrix(color, strength, num_points, has_color, has_strength)

    kwargs = dict(
        pos=torch.from_numpy(pos),
        x=torch.from_numpy(x),
        x_features_names=list(X_FEATURE_NAMES),
        idx_in_original_cloud=np.arange(num_points, dtype=np.int32),
    )
    if not has_color:
        kwargs["color_mask"] = torch.ones(num_points, dtype=torch.bool)
    if not has_strength:
        kwargs["strength_mask"] = torch.ones(num_points, dtype=torch.bool)

    label_path = osp.join(scene_dir, f"{label_key}.npy")
    if osp.isfile(label_path):
        # Pointcept's own preprocessing already remapped labels to contiguous train
        # ids -- no remapping here (mirrors pointcept_npy.py's segment.npy handling).
        y = np.load(label_path).reshape(-1).astype(np.int64, copy=False)
        kwargs["y"] = torch.from_numpy(y)

    return Data(**kwargs)


class DownstreamNpyDataset(Dataset):
    """Dataset over a Pointcept-preprocessed downstream ``{data_root}/{split_dir}/*/`` layout."""

    def __init__(
        self,
        data_root: str,
        split_dir: str,
        has_color: bool = True,
        has_strength: bool = True,
        label_key: str = "segment",
        is_eval: bool = False,
        tile_width: Number = 50,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
        transform: Optional[Callable] = None,
        scene_dirs: Optional[Sequence[str]] = None,
    ):
        self.data_root = data_root
        self.split_dir = split_dir
        self.has_color = has_color
        self.has_strength = has_strength
        self.label_key = label_key
        self.pre_filter = pre_filter
        self.transform = transform
        self.entries: List[SceneEntry] = build_scene_entries(
            data_root,
            split_dir,
            is_eval=is_eval,
            tile_width=tile_width,
            subtile_width=subtile_width,
            subtile_overlap=subtile_overlap,
            scene_dirs=scene_dirs,
        )
        if not self.entries:
            raise FileNotFoundError(
                f"No scene directories found under {osp.join(data_root, split_dir)}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Optional[Data]:
        scene_dir, subtile_index = self.entries[idx]
        data = load_downstream_scene(
            scene_dir,
            has_color=self.has_color,
            has_strength=self.has_strength,
            label_key=self.label_key,
        )
        patch_id = osp.basename(osp.normpath(scene_dir))
        data.patch_id = patch_id
        if subtile_index is not None:
            data.subtile_index = subtile_index

        if self.pre_filter and self.pre_filter(data):
            return None

        if self.transform:
            data = self.transform(data)

        if not data or (self.pre_filter and self.pre_filter(data)):
            return None

        # Re-attach in case a transform dropped the python string attribute.
        data.patch_id = patch_id
        return data
