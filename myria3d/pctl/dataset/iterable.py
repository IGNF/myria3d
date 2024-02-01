from numbers import Number
from typing import Callable, Optional

import torch
from numpy.typing import ArrayLike
from torch.utils.data.dataset import IterableDataset
from torch_geometric.data import Data

from myria3d.pctl.dataset.utils import (
    pre_filter_below_n_points,
    split_cloud_into_samples,
)
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform


class InferenceDataset(IterableDataset):
    """Iterable dataset to load samples from a single las file."""

    def __init__(
        self,
        las_file: str,
        points_pre_transform: Callable[[ArrayLike], Data] = lidar_hd_pre_transform,
        pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
        transform: Optional[Callable[[Data], Data]] = None,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
    ):
        self.las_file = las_file

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter
        self.transform = transform

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap = subtile_overlap

    def __iter__(self):
        return self.get_iterator()

    def get_iterator(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""
        for idx_in_original_cloud, sample_points in split_cloud_into_samples(
            self.las_file,
            self.tile_width,
            self.subtile_width,
            self.subtile_overlap,
        ):
            sample_data = self.points_pre_transform(sample_points)
            sample_data["x"] = torch.from_numpy(sample_data["x"])
            sample_data["y"] = torch.LongTensor(
                sample_data["y"]
            )  # Need input classification for DropPointsByClass
            sample_data["pos"] = torch.from_numpy(sample_data["pos"])
            sample_data["cluster_id"] = torch.from_numpy(sample_data["cluster_id"])
            # for final interpolation - should be kept as a np.ndarray to be batched as a list later.
            sample_data["idx_in_original_cloud"] = idx_in_original_cloud

            if self.pre_filter and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue

            if self.transform:
                sample_data = self.transform(sample_data)

            if sample_data is None:
                continue

            if self.pre_filter and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue

            yield sample_data
