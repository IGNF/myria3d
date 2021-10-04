""" We define two datasets:
  - TrainDataset with random selection of a sub-tile
  - ValDataset = TestDataset with an exhaustive parsing of the sub-tiles.
"""


import copy
import random
from itertools import chain, cycle

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data

from semantic_val.datamodules.datasets.lidar_utils import get_all_subtile_centers, load_las_data
from semantic_val.utils import utils

log = utils.get_logger(__name__)


class LidarTrainDataset(Dataset):
    def __init__(
        self,
        files,
        transform=None,
        target_transform=None,
        subtile_width_meters: float = 100,
    ):
        self.files = files
        self.transform = transform
        self.target_transform = target_transform
        self.subtile_width_meters = subtile_width_meters
        self.in_memory_filepath = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get a subtitle from indexed las file, and apply the transforms specified in datamodule."""
        filepath = self.files[idx]
        data = self.get_cloud_data(filepath)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            data = self.target_transform(data)

        return data

    def get_cloud_data(self, filepath):
        """Get already in-memory cloud data or load it from disk."""
        if self.in_memory_filepath == filepath:
            data = self.data
        else:
            log.debug(f"Loading train file: {filepath}")
            data = load_las_data(filepath)
            self.in_memory_filepath = filepath
            self.data = data
        return data


class LidarValDataset(IterableDataset):
    def __init__(
        self,
        files,
        transform=None,
        target_transform=None,
        subtile_overlap: float = 0,
        subtile_width_meters: float = 100,
    ):
        self.files = files
        self.transform = transform
        self.target_transform = target_transform
        self.subtile_overlap = subtile_overlap
        self.subtile_width_meters = subtile_width_meters

    def process_data(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""

        for filepath in self.files:
            tile_data = load_las_data(filepath)
            centers = get_all_subtile_centers(
                tile_data,
                subtile_width_meters=self.subtile_width_meters,
                subtile_overlap=self.subtile_overlap,
            )
            for tile_data.current_subtile_center in centers:
                if self.transform:
                    data = self.transform(tile_data)
                if self.target_transform:
                    data = self.target_transform(data)
                yield data

    def __iter__(self):
        return self.process_data()


LidarToyTestDataset = LidarValDataset
