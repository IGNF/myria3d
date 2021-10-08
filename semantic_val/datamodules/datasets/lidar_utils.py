import os.path as osp
from pathlib import Path
from typing import List, Union

import laspy
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset

import shapefile
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


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


def collate_fn(data_list: List[Data]) -> Batch:
    """Collate list of Data elements, to be used in DataLoader.
    From: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dense_data_loader.html?highlight=collate_fn"""
    batch = Batch()

    # 1: add everything as list of non-Tensor object to facilitate adding new attributes.
    for key in data_list[0].keys:
        batch[key] = [data[key] for data in data_list]

    # 2: define relevant Tensor in long PyG format.
    # batch.x = torch.from_numpy(np.concatenate([data.x for data in data_list]))
    # batch.pos = torch.from_numpy(np.concatenate([data.pos for data in data_list]))
    # batch.y = torch.from_numpy(np.concatenate([data.y for data in data_list]))
    batch.pos = torch.cat([data.pos for data in data_list])
    batch.origin_pos = torch.cat([data.origin_pos for data in data_list])
    batch.x = torch.cat([data.x for data in data_list])
    batch.y = torch.cat([data.y for data in data_list])
    batch.batch = torch.from_numpy(
        np.concatenate(
            [
                np.full(shape=len(data.y), fill_value=i)
                for i, data in enumerate(data_list)
            ]
        )
    )
    return batch


## Utils for data preparation


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
