from numbers import Number
import torch
from torch.utils.data.dataset import IterableDataset
from myria3d.pctl.dataset.utils import SHAPE_TYPE, split_cloud_into_samples


class InferenceDataset(IterableDataset):
    """No need for HDF5 for Inference."""

    def __init__(
        self,
        las_file: str,
        points_pre_transform=None,
        pre_filter=None,
        transform=None,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        subtile_shape: SHAPE_TYPE = "square",
    ):
        self.las_file = las_file

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter
        self.transform = transform

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_shape = subtile_shape
        self.subtile_overlap = subtile_overlap

    def __iter__(self):
        return self.get_iterator()

    def get_iterator(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""
        for idx_in_original_cloud, sample_points in split_cloud_into_samples(
            self.las_file,
            self.tile_width,
            self.subtile_width,
            self.subtile_shape,
            self.subtile_overlap,
        ):
            sample_data = self.points_pre_transform(sample_points)
            sample_data["x"] = torch.from_numpy(sample_data["x"])
            sample_data["y"] = torch.from_numpy(sample_data["y"])
            sample_data["pos"] = torch.from_numpy(sample_data["pos"])
            # for final interpolation
            sample_data["idx_in_original_cloud"] = torch.from_numpy(
                idx_in_original_cloud
            )

            if self.pre_filter is not None and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue

            if self.transform:
                sample_data = self.transform(sample_data)

            if self.pre_filter is not None and self.pre_filter(sample_data):
                # e.g. not enough points in this receptive field.
                continue

            yield sample_data
