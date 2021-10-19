import math
import os.path as osp
from pathlib import Path
from typing import Callable, List
import random

import laspy
import numpy as np
import pandas as pd
import shapefile
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform, Center

from semantic_val.utils import utils

log = utils.get_logger(__name__)


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
    )
    x = np.asarray(
        [
            las.intensity,
            las.return_num,
            las.num_returns,
        ],
        dtype=np.float32,
    )
    pos = pos.transpose()
    x = x.transpose()
    y = las.classification.astype(np.int)
    tile_id = Path(filepath).stem

    return Data(
        pos=pos,
        x=x,
        y=y,
        filepath=filepath,
        tile_id=tile_id,
    )


def get_random_subtile_center(data: Data, subtile_width_meters: float = 100.0):
    """
    Randomly select x/y pair (in meters) as potential center of a square subtile of original tile
    (whose x and y coordinates are in meters and in 0m-1000m range).
    """
    half_subtile_width_meters = subtile_width_meters / 2
    low = data.pos[:, :2].min(0) + half_subtile_width_meters
    high = data.pos[:, :2].max(0) - half_subtile_width_meters

    subtile_center_xy = np.random.uniform(low, high)

    return subtile_center_xy


def get_tile_center(data: Data, subtile_width_meters: float = 100.0):
    """
    Randomly select x/y pair (in meters) as potential center of a square subtile of original tile
    (whose x and y coordinates are in meters and in 0m-1000m range).
    """
    half_subtile_width_meters = subtile_width_meters / 2
    low = data.pos[:, :2].min(0) + half_subtile_width_meters
    high = data.pos[:, :2].max(0) - half_subtile_width_meters

    subtile_center_xy = (high + low) / 2

    return subtile_center_xy


def get_all_subtile_centers(
    data: Data, subtile_width_meters: float = 100.0, subtile_overlap: float = 0
):
    """Get centers of square subtiles of specified width, assuming rectangular form of input cloud."""

    half_subtile_width_meters = subtile_width_meters / 2
    low = data.pos[:, :2].min(0) + half_subtile_width_meters
    high = data.pos[:, :2].max(0) - half_subtile_width_meters + 1
    centers = [
        (x, y)
        for x in np.arange(
            start=low[0], stop=high[0], step=subtile_width_meters - subtile_overlap
        )
        for y in np.arange(
            start=low[1], stop=high[1], step=subtile_width_meters - subtile_overlap
        )
    ]
    random.shuffle(centers)
    return centers


def get_subsampling_mask(input_size: int, subsampling_size: int):
    """Get a mask to select subsampling_size elements from an iterable of specified size, with replacement if needed."""

    if input_size >= subsampling_size:
        sampled_points_idx = np.random.choice(
            input_size, subsampling_size, replace=False
        )
    else:
        sampled_points_idx = np.concatenate(
            [
                np.arange(input_size),
                np.random.choice(
                    input_size, subsampling_size - input_size, replace=True
                ),
            ]
        )
    return sampled_points_idx


def get_subtile_data(
    data: Data,
    subtile_center_xy,
    subtile_width_meters: float = 100.0,
):
    """Extract tile points and labels around a subtile center using Chebyshev distance, in meters."""
    subtile_data = data.clone()

    chebyshev_distance = np.max(
        np.abs(subtile_data.pos[:, :2] - subtile_center_xy), axis=1
    )
    mask = chebyshev_distance <= (subtile_width_meters / 2)

    subtile_data.pos = subtile_data.pos[mask]
    subtile_data.x = subtile_data.x[mask]
    subtile_data.y = subtile_data.y[mask]

    return subtile_data


# Data transforms


class CustomCompose(BaseTransform):
    """Composes several transforms together.
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

    def __repr__(self):
        args = ["    {},".format(transform) for transform in self.transforms]
        return "{}([\n{}\n])".format(self.__class__.__name__, "\n".join(args))


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

        MAX_TRY_IN_TRAIN_MODE = 25

        for try_i in range(1, MAX_TRY_IN_TRAIN_MODE + 1):
            if self.method == "random":
                center = get_random_subtile_center(
                    data, subtile_width_meters=self.subtile_width_meters
                )
            elif self.method == "predefined":
                center = data.current_subtile_center
            else:
                raise f"Undefined method argument: {self.method}"

            subtile_data = get_subtile_data(
                data,
                center,
                subtile_width_meters=self.subtile_width_meters,
            )
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


class ToTensor(BaseTransform):
    r"""Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys=["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


# TODO: find a better naming convention for copy-full vs copy-subsample vs normalized-full vs normalized-subsample...
class MakeCopyOfPosAndY(BaseTransform):
    r"""Make a copy of the full cloud's positions and labels, for final interpolation."""

    def __call__(self, data: Data):
        data["pos_copy"] = data["pos"].clone()
        data["y_copy"] = data["y"].clone()
        return data


class FixedPointsPosXY(BaseTransform):
    r"""Samples a fixed number of :obj:`num` positions, targets and features from a point cloud.
    # From https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html#FixedPoints
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


class NormalizeFeatures(BaseTransform):
    r"""Scale features in 0-1 range."""

    def __call__(self, data: Data):
        INTENSITY_IDX = 0
        RETURN_NUM_IDX = 1
        NUM_RETURN_IDX = 2

        INTENSITY_MAX = 32768.0
        RETURN_NUM_MAX = 7

        data["x"][:, INTENSITY_IDX] = data["x"][:, INTENSITY_IDX] / INTENSITY_MAX
        data["x"][:, RETURN_NUM_IDX] = (data["x"][:, RETURN_NUM_IDX] - 1) / (
            RETURN_NUM_MAX - 1
        )
        data["x"][:, NUM_RETURN_IDX] = (data["x"][:, NUM_RETURN_IDX] - 1) / (
            RETURN_NUM_MAX - 1
        )
        return data


class CustomNormalizeScale(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`."""

    def __init__(self):
        self.center = Center()

    def __call__(self, data):
        data = self.center(data)

        scale = (1 / data.pos[:, :2].abs().max()) * 0.999999
        data.pos[:, :2] = data.pos[:, :2] * scale

        z_scale = 100.0
        data.pos[:, 2] = (data.pos[:, 2] - data.pos[:, 2].min()) / 100

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
    """Collate list of Data elements, to be used in DataLoader.
    From: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dense_data_loader.html?highlight=collate_fn"""
    batch = Batch()
    data_list = filter(lambda x: x is not None, data_list)

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


# Prepare dataset


def get_shapefile_records_df(sf):
    """Get the shapefile records as a pd.DataFrame."""
    records = sf.records()
    fields = [x[0] for x in sf.fields][1:]
    df = pd.DataFrame(columns=fields, data=records)
    return df


def get_metadata_df_from_shapefile(filepath):
    """Get the shapefile records of tiles metadata as a formated pd.DataFrame."""
    sf = shapefile.Reader(filepath)
    df = get_shapefile_records_df(sf)
    df = df.sort_values(by="file")
    int_col = ["port", "nb_vehicul", "nb_bati", "nb_veget", "nb_autre"]
    df[int_col] = df[int_col].astype(int)
    return df


def get_split_df_of_202110_building_val(df, train_frac=0.8):
    """
    Dataset name: "202110_building_val"
    From the formated pd.DataFrame of tiles metadata (output by get_metadata_df_from_shapefile)
    stratify the dataset based on geographical layer and port areas.
    """

    train_tiles = []
    val_tiles = []
    test_tiles = []

    df_to_split = df[df.layer == "71_polygo"]
    tesval_n = int((1 - train_frac) * len(df_to_split))
    train_n = len(df_to_split) - tesval_n

    train, val, test = train, validate, test = np.split(
        df_to_split.sample(frac=1, random_state=0),
        [train_n + 1, train_n + int(tesval_n / 2) + 1],
    )
    train_tiles.append(train)
    val_tiles.append(val)
    test_tiles.append(test)
    # print(train.shape, test.shape, val.shape)

    df_to_split = df[df.layer == "forca_polygo"]
    tesval_n = int((1 - train_frac) * len(df_to_split))
    train_n = len(df_to_split) - tesval_n
    train, val, test = train, validate, test = np.split(
        df_to_split.sample(frac=1, random_state=0),
        [train_n, train_n + int(tesval_n / 2)],
    )
    train_tiles.append(train)
    val_tiles.append(val)
    test_tiles.append(test)
    # print(train.shape, test.shape, val.shape)

    df_to_split = df[df.layer == "la_grande_motte_polygo"]

    ports = df_to_split.loc[df_to_split.port == 1]
    assert len(ports) == 2
    test_tiles.append(ports.iloc[0:1])
    train_tiles.append(ports.iloc[1:2])

    noports = df_to_split[df_to_split.port == 0].sample(frac=1, random_state=0)
    assert len(noports) == 6
    train_tiles.append(noports.iloc[:4])
    test_tiles.append(noports.iloc[4:5])
    val_tiles.append(noports.iloc[5:6])

    train_tiles = pd.concat(train_tiles)
    test_tiles = pd.concat(test_tiles)
    val_tiles = pd.concat(val_tiles)

    train_tiles["split"] = "train"
    val_tiles["split"] = "val"
    test_tiles["split"] = "test"

    assert len(train_tiles) == 120
    assert len(val_tiles) == 15
    assert len(test_tiles) == 15

    df_split = pd.concat([train_tiles, val_tiles, test_tiles])

    return df_split


def create_full_filepath_column(df, dirpath):
    """Append dirpath as a suffix to file column"""
    df["file_path"] = df["file"].apply(lambda stem: osp.join(dirpath, stem))
    return df
