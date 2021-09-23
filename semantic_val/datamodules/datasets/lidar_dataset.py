import copy
import random
from itertools import chain, cycle

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data import Data

from semantic_val.datamodules.datasets.lidar_transforms import (
    get_all_subtile_centers,
    get_random_subtile_center,
    get_subtile_data,
    load_las_file,
)


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

        self.in_memory_filename = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get a subtitle from indexed las file, and apply the transforms specified in datamodule."""
        filename = self.files[idx]

        if self.in_memory_filename != filename:
            cloud, labels = load_las_file(filename)
            self.in_memory_filename = filename
            self.in_memory_cloud = cloud
            self.in_memory_labels = labels
        else:
            cloud = self.in_memory_cloud
            labels = self.in_memory_labels

        # TODO: move into transfrom to abstract more.
        center = get_random_subtile_center(cloud, subtile_width_meters=self.subtile_width_meters)
        cloud, labels = get_subtile_data(
            copy.deepcopy(cloud),
            copy.deepcopy(labels),
            center,
            subtile_width_meters=self.subtile_width_meters,
        )

        if self.transform:
            cloud = self.transform(cloud)

        if self.target_transform:
            labels = self.target_transform(labels)

        # TODO: consider moving up the use of Data structure for the transforms to be used on it.
        data = Data(pos=cloud[:, :3], x=cloud, y=labels)

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

        self.in_memory_filename = None

    def process_data(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""

        for filename in self.files:
            cloud_full, labels_full = load_las_file(filename)
            centers = get_all_subtile_centers(
                cloud_full,
                subtile_width_meters=self.subtile_width_meters,
                subtile_overlap=self.subtile_overlap,
            )
            for center in centers:
                # TODO: move into transfrom to abstract more.
                cloud, labels = get_subtile_data(
                    cloud_full,
                    labels_full,
                    center,
                    subtile_width_meters=self.subtile_width_meters,
                )

                if self.transform:
                    cloud = self.transform(cloud)

                if self.target_transform:
                    labels = self.target_transform(labels)

                # TODO: consider moving up the use of Data structure for the transforms to be used on it.
                data = Data(pos=cloud[:, :3], x=cloud, y=labels)

                yield data

    def __iter__(self):
        return self.process_data()


LidarToyTestDataset = LidarValDataset
