# pylint: disable
import math
from pathlib import Path
from typing import Callable, List

import laspy
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

from semantic_val.utils import utils

log = utils.get_logger(__name__)

# CONSTANTS

# Warning: be sure that this oder matches the one in load_las_data.
COLORS_NAMES = ["red", "green", "blue", "nir"]
X_FEATURES_NAMES = [
    "intensity",
    "return_num",
    "num_returns",
] + COLORS_NAMES

INTENSITY_MAX = 32768.0
COLORS_MAX = 255 * 256
MAX_TRY_IN_TRAIN_MODE = 25
RETURN_NUM_MAX = 7

HALF_UNIT = 0.5
UNIT = 1

# DATA LOADING


def load_las_data(filepath):
    """Load a cloud of points and its labels. base shape: [n_points, n_features].
    Warning: las.x is in meters, las.X is in centimeters.
    """
    las = laspy.read(filepath)
    pos = np.asarray(
        [
            las.x,
            las.y,
            las.z,
        ],
        dtype=np.float32,
    ).transpose()
    x = np.asarray(
        [las[x_name] for x_name in X_FEATURES_NAMES],
        dtype=np.float32,
    ).transpose()
    y = las.classification.astype(np.int)
    tile_id = Path(filepath).stem

    return Data(
        pos=pos,
        x=x,
        y=y,
        filepath=filepath,
        tile_id=tile_id,
        x_features_names=X_FEATURES_NAMES,
    )


# DATA TRANSFORMS


class CustomCompose(BaseTransform):
    """
    Composes several transforms together.
    Edited to bypass downstream transforms if None is returned by a transform.

    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
                data = filter(lambda x: x is not None, data)
            else:
                data = transform(data)
                if data is None:
                    return None
        return data


class SelectSubTile(BaseTransform):
    r"""Select a square subtile from original tile"""

    def __init__(
        self,
        subtile_width_meters: float = 100.0,
        method=["deterministic", "predefined", "random"],
    ):
        self.subtile_width_meters = subtile_width_meters
        self.method = method

    def __call__(self, data: Data):

        for try_i in range(1, MAX_TRY_IN_TRAIN_MODE + 1):
            if self.method == "random":
                center = self.get_random_subtile_center(data)
            elif self.method == "predefined":
                center = data.current_subtile_center
            else:
                raise f"Undefined method argument: {self.method}"

            subtile_data = self.get_subtile_data(data, center)
            if len(subtile_data.pos) > 0:
                return subtile_data
            else:
                log.debug(f"No points in {data.filepath} around xy = {str(center)}")
                if self.method == "random":
                    if try_i < MAX_TRY_IN_TRAIN_MODE:
                        log.debug(
                            f"Trying another center... [{try_i+1}/{MAX_TRY_IN_TRAIN_MODE}]"
                        )
                        continue
                    else:
                        log.debug(
                            f"Skipping extraction after {MAX_TRY_IN_TRAIN_MODE} independant try."
                        )
                        return None
                else:
                    log.debug("Ignoring this subtile.")
                    return None

    def get_random_subtile_center(self, data: Data):
        """
        Randomly select x/y pair (in meters) as potential center of a square subtile of original tile
        (whose x and y coordinates are in meters and in 0m-1000m range).
        """
        half_subtile_width_meters = self.subtile_width_meters / 2
        low = data.pos[:, :2].min(0) + half_subtile_width_meters
        high = data.pos[:, :2].max(0) - half_subtile_width_meters

        subtile_center_xy = np.random.uniform(low, high)

        return subtile_center_xy

    def get_subtile_data(self, data: Data, subtile_center_xy):
        """Extract tile points and labels around a subtile center using Chebyshev distance, in meters."""
        subtile_data = data.clone()

        chebyshev_distance = np.max(
            np.abs(subtile_data.pos[:, :2] - subtile_center_xy), axis=1
        )
        mask = chebyshev_distance <= (self.subtile_width_meters / 2)

        subtile_data.pos = subtile_data.pos[mask]
        subtile_data.x = subtile_data.x[mask]
        subtile_data.y = subtile_data.y[mask]

        return subtile_data


class ToTensor(BaseTransform):
    r"""Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys=["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


class MakeCopyOfPosAndY(BaseTransform):
    r"""Make a copy of the full cloud's positions and labels, for inference interpolation."""

    def __call__(self, data: Data):
        data["pos_copy"] = data["pos"].clone()
        data["y_copy"] = data["y"].clone()
        return data


class FixedPointsPosXY(BaseTransform):
    r"""
    Samples a fixed number of points from a point cloud.
    Modified to preserve specific attributes of the data for inference interpolation, from
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html#FixedPoints
    """

    def __init__(self, num, replace=True, allow_duplicates=False):
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def __call__(self, data: Data, keys=["x", "pos", "y"]):
        num_nodes = data.num_nodes

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[: self.num]
        else:
            choice = torch.cat(
                [
                    torch.randperm(num_nodes)
                    for _ in range(math.ceil(self.num / num_nodes))
                ],
                dim=0,
            )[: self.num]

        for key in keys:
            data[key] = data[key][choice]

        return data

    def __repr__(self):
        return "{}({}, replace={})".format(
            self.__class__.__name__, self.num, self.replace
        )


class MakeCopyOfSampledPos(BaseTransform):
    """Make a copy of the unormalized positions of subsampled points."""

    def __call__(self, data: Data):
        data["pos_copy_subsampled"] = data["pos"].clone()
        return data


class RandomTranslateFeatures(BaseTransform):
    r"""
    Randomly translate the (unnormalized) features values.

    Intensity: random translate by rel_translation * max
    Colors (RGB): random translate by rel_translation * max
    Number of returns: +1/+0/-1 with equal probability
    Return number: +1/+0/-1 with equal probability, max-clamped by number of returns.
    """

    def __call__(self, data: Data, rel_translation: float = 0.02):

        x = data.x
        (n, _) = x.size()

        translation = rel_translation * INTENSITY_MAX
        intensity_idx = data.x_features_names.index("intensity")
        delta = x[:, intensity_idx].new_empty(n).uniform_(-translation, translation)
        x[:, intensity_idx] = x[:, intensity_idx] + delta
        x[:, intensity_idx] = x[:, intensity_idx].clamp(min=0, max=INTENSITY_MAX)

        translation = rel_translation * COLORS_MAX
        COLORS_IDX = [
            data.x_features_names.index(color_name) for color_name in COLORS_NAMES
        ]
        for color_idx in COLORS_IDX:
            delta = x[:, color_idx].new_empty(n).uniform_(-translation, translation)
            x[:, color_idx] = x[:, color_idx] + delta
            x[:, color_idx] = x[:, color_idx].clamp(min=0, max=COLORS_MAX)

        num_return_idx = data.x_features_names.index("num_returns")
        delta = x[:, num_return_idx].new_empty(n).random_(-1, 2)
        x[:, num_return_idx] = x[:, num_return_idx] + delta
        x[:, num_return_idx] = x[:, num_return_idx].clamp(min=1, max=RETURN_NUM_MAX)

        return_num_idx = data.x_features_names.index("return_num")
        delta = x[:, return_num_idx].new_empty(n).random_(-1, 2)
        x[:, return_num_idx] = x[:, return_num_idx] + delta
        x[:, return_num_idx] = x[:, return_num_idx].clamp(min=1)
        x[:, return_num_idx] = torch.min(x[:, return_num_idx], x[:, num_return_idx])

        return data


class CustomNormalizeFeatures(BaseTransform):
    r"""Scale features in 0-1 range."""

    def __call__(self, data: Data):

        intensity_idx = data.x_features_names.index("intensity")
        data.x[:, intensity_idx] = data.x[:, intensity_idx] / INTENSITY_MAX - HALF_UNIT

        colors_idx = [
            data.x_features_names.index(color_name) for color_name in COLORS_NAMES
        ]
        for color_idx in colors_idx:
            data.x[:, color_idx] = data.x[:, color_idx] / COLORS_MAX - HALF_UNIT

        return_num_idx = data.x_features_names.index("return_num")
        data.x[:, return_num_idx] = (data.x[:, return_num_idx] - UNIT) / (
            RETURN_NUM_MAX - UNIT
        ) - HALF_UNIT
        num_return_idx = data.x_features_names.index("num_returns")
        data.x[:, num_return_idx] = (data.x[:, num_return_idx] - UNIT) / (
            RETURN_NUM_MAX - UNIT
        ) - HALF_UNIT

        return data


class CustomNormalizeScale(BaseTransform):
    r"""Normalizes node positions to the interval (-1, 1)."""

    def __init__(self, z_scale: float = 100.0):
        self.z_scale = z_scale
        pass

    def __call__(self, data):

        xy_scale = (1 / data.pos[:, :2].abs().max()) * 0.999999
        data.pos[:, :2] = data.pos[:, :2] * xy_scale

        data.pos[:, 2] = (data.pos[:, 2] - data.pos[:, 2].min()) / self.z_scale

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class MakeBuildingTargets(BaseTransform):
    """
    Pass from multiple classes to simpler Building/Non-Building labels.
    Initial classes: [  1,   2,   6 (detected building, no validation),  19 (valid building),  20 (surdetection, unspecified),
    21 (building, forgotten), 104, 110 (surdetection, others), 112 (surdetection, vehicule), 114 (surdetection, others), 115 (surdetection, bridges)]
    Final classes: 0 (non-building), 1 (building)
    Applied on both unsampled and subsampled labels (only because target_transforms are called after transforms)
    """

    def __call__(self, data: Data, keys: List[str] = ["y", "y_copy"]):
        for key in keys:
            data[key] = self.make_building_targets(data[key])
        return data

    def make_building_targets(self, y):
        buildings_idx = (y == 19) | (y == 21) | (y == 6)
        y[buildings_idx] = 1
        y[~buildings_idx] = 0
        return y


def collate_fn(data_list: List[Data]) -> Batch:
    """
    Batch Data objects from a list, to be used in DataLoader. Modified from:
    https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dense_data_loader.html?highlight=collate_fn
    """
    batch = Batch()
    data_list = list(filter(lambda x: x is not None, data_list))

    # 1: add everything as list of non-Tensor object to facilitate adding new attributes.
    for key in data_list[0].keys:
        batch[key] = [data[key] for data in data_list]

    # 2: define relevant Tensor in long PyG format.
    keys_to_long_format = ["pos", "x", "y", "pos_copy", "y_copy", "pos_copy_subsampled"]
    for key in keys_to_long_format:
        batch[key] = torch.cat([data[key] for data in data_list])

    # 3. Create a batch index
    batch.batch_x = torch.from_numpy(
        np.concatenate(
            [
                np.full(shape=len(data["y"]), fill_value=i)
                for i, data in enumerate(data_list)
            ]
        )
    )
    batch.batch_y = torch.from_numpy(
        np.concatenate(
            [
                np.full(shape=len(data["y_copy"]), fill_value=i)
                for i, data in enumerate(data_list)
            ]
        )
    )
    batch.batch_size = len(data_list)
    return batch
