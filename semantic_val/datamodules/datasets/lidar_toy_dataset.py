import copy
import random
from itertools import chain, cycle

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from semantic_val.datamodules.datasets.lidar_transforms import (
    get_random_subtile_center,
    get_subtile_data,
    load_las_file,
)


class LidarToyTrainDataset(Dataset):
    def __init__(
        self,
        files,
        transform=None,
        target_transform=None,
        input_cloud_size: int = 200000,
        subtile_width_meters: float = 100,
    ):
        self.files = files
        self.transform = transform
        self.target_transform = target_transform

        self.input_cloud_size = input_cloud_size
        self.subtile_width_meters = subtile_width_meters

        self.in_memory_filename = None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """Get a subtitle from indexed las file, and apply the transforms specified in datamodule."""
        filename = self.files[idx]

        # Avoid consecutive loading of same file if subsequent in files list.
        # TODO: assure this is useful by sorting duplicated files in a clever fashion.
        cloud, labels = load_las_file(filename)
        # if self.in_memory_filename != filename:
        #     self.in_memory_filename = filename
        #     self.in_memory_cloud = cloud
        #     self.in_memory_labels = labels
        # else:
        #     # TODO: check for unwanted inplace modifications.
        #     cloud = self.in_memory_cloud
        #     labels = self.in_memory_labels
        center = get_random_subtile_center(cloud, subtile_width_meters=self.subtile_width_meters)
        cloud, labels = get_subtile_data(
            copy.deepcopy(cloud),
            copy.deepcopy(labels),
            center,
            input_cloud_size=self.input_cloud_size,
            subtile_width_meters=self.subtile_width_meters,
        )

        if self.transform:
            cloud = self.transform(cloud)

        if self.target_transform:
            labels = self.target_transform(labels)

        return cloud, labels


class LidarToyValDataset(IterableDataset):
    def __init__(
        self,
        files,
        transform=None,
        target_transform=None,
        input_cloud_size: int = 200000,
        # tile_width_meters: float = 1000.0,  # TODO: add as a global, fixed parameter in default configs.
        subtile_width_meters: float = 100,
    ):
        self.files = files
        self.transform = transform
        self.target_transform = target_transform

        self.input_cloud_size = input_cloud_size
        # self.tile_width_meters = tile_width_meters
        self.subtile_width_meters = subtile_width_meters

        self.in_memory_filename = None

    def process_data(self):
        for filename in self.files:
            cloud_full, labels_full = load_las_file(filename)
            centers = [(50, 50), (100, 100), (150, 150), (200, 200)]
            for center in centers:
                # get the data from the in memory las
                cloud, labels = get_subtile_data(
                    cloud_full,
                    labels_full,
                    center,
                    input_cloud_size=self.input_cloud_size,
                    subtile_width_meters=self.subtile_width_meters,
                )

                if self.transform:
                    cloud = self.transform(cloud)

                if self.target_transform:
                    labels = self.target_transform(labels)

                yield cloud, labels

    def __iter__(self):
        return self.process_data()


LidarToyTestDataset = LidarToyValDataset
