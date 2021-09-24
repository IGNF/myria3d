from pathlib import Path
from typing import List, Union

import laspy
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Data, Dataset


def load_las_file(filename):
    """Load a cloud of points and its labels. We transpose to have cloud with shape [n_points, n_features]."""
    las = laspy.read(filename)
    cloud = np.asarray(
        [
            las.x - las.x.min(),
            las.y - las.y.min(),
            las.z,
            las.intensity,
            las.return_num,
            las.num_returns,
        ],
        dtype=np.float32,
    )
    cloud = cloud.transpose()
    labels = las.classification.astype(np.int)
    return cloud, labels


def load_las_data(filepath):
    cloud, labels = load_las_file(filepath)
    # TODO: remove the .copy if not useful.
    tile_id = Path(filepath).stem
    data = Data(pos=cloud[:, :3].copy(), x=cloud, y=labels, filepath=filepath, tile_id=tile_id)
    return data


def get_random_subtile_center(data: Data, subtile_width_meters: float = 100.0):
    """
    Randomly select x/y pair (in meters) as potential center of a square subtile of original tile
    (whose x and y coordinates are in meters and in 0m-1000m range).
    """
    half_subtile_width_meters = subtile_width_meters / 2
    low = data.x[:, :2].min(0) + half_subtile_width_meters
    high = data.x[:, :2].max(0) - half_subtile_width_meters

    subtile_center_xy = np.random.uniform(low, high)

    return subtile_center_xy


def get_all_subtile_centers(
    data: Data, subtile_width_meters: float = 100.0, subtile_overlap: float = 0
):
    """Get centers of square subtiles of specified width, assuming rectangular form of input cloud."""

    half_subtile_width_meters = subtile_width_meters / 2
    low = data.x[:, :2].min(0) + half_subtile_width_meters
    high = data.x[:, :2].max(0) - half_subtile_width_meters + 1
    centers = [
        (x, y)
        for x in np.arange(start=low[0], stop=high[0], step=subtile_width_meters - subtile_overlap)
        for y in np.arange(start=low[1], stop=high[1], step=subtile_width_meters - subtile_overlap)
    ]
    return centers


def get_subsampling_mask(input_size: int, subsampling_size: int):
    """Get a mask to select subsampling_size elements from an iterable of specified size, with replacement if needed."""

    if input_size >= subsampling_size:
        sampled_points_idx = np.random.choice(input_size, subsampling_size, replace=False)
    else:
        sampled_points_idx = np.concatenate(
            [
                np.arange(input_size),
                np.random.choice(input_size, subsampling_size - input_size, replace=True),
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

    chebyshev_distance = np.max(np.abs(subtile_data.pos[:, :2] - subtile_center_xy), axis=1)
    mask = chebyshev_distance < (subtile_width_meters / 2)

    subtile_data.x = subtile_data.x[mask]
    subtile_data.pos = subtile_data.pos[mask]
    subtile_data.y = subtile_data.y[mask]

    return subtile_data


def transform_labels_for_building_segmentation(data: Data):
    """
    Pass from multiple classes to simpler Building/Non-Building labels.
    Initial classes: [  1,   2,   6 (detected building, no validation),  19 (valid building),  20 (surdetection, unspecified),
    21 (building, forgotten), 104, 110 (surdetection, others), 112 (surdetection, vehicule), 114 (surdetection, others), 115 (surdetection, bridges)]
    Final classes: 0 (non-building), 1 (building)
    """
    buildings = (data.y == 19) | (data.y == 21) | (data.y == 6)
    data.y[buildings] = 1
    data.y[~buildings] = 0
    return data


def collate_fn(data_list: List[Data]) -> Batch:
    """Collate list of Data elements, to be used in DataLoader.
    From: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dense_data_loader.html?highlight=collate_fn"""
    batch = Batch()

    # 1: add everything as list of non-Tensor object to facilitate adding new attributes.
    for key in data_list[0].keys:
        batch[key] = [data[key] for data in data_list]

    # 2: define relevant Tensor in long PyG format.
    batch.x = torch.from_numpy(np.concatenate([data.x for data in data_list]))
    batch.pos = torch.from_numpy(np.concatenate([data.pos for data in data_list]))
    batch.y = torch.from_numpy(np.concatenate([data.y for data in data_list]))
    batch.batch = torch.from_numpy(
        np.concatenate(
            [np.full(shape=len(data.y), fill_value=i) for i, data in enumerate(data_list)]
        )
    )
    return batch
