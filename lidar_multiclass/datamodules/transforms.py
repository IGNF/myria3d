# pylint: disable
import math
from enum import Enum
from typing import Callable, Dict, List

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

from lidar_multiclass.utils import utils

log = utils.get_logger(__name__)


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


class EmptySubtileFilter(BaseTransform):
    r"""Filter out almost empty subtiles"""

    def __call__(self, data: Data, min_num_points_subtile: int = 50):
        if len(data["x"]) < min_num_points_subtile:
            return None
        return data


class ToTensor(BaseTransform):
    r"""Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys=["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


class MakeCopyOfPos(BaseTransform):
    r"""Make a copy of the full cloud's positions and labels, for inference interpolation."""

    def __call__(self, data: Data):
        data["pos_copy"] = data["pos"].clone()
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


class StandardizeFeatures(BaseTransform):
    r"""
    Scale features in 0-1 range.
    Additionnaly : use reserved -0.75 value for occluded points colors(normal range is -0.5 to 0.5).
    """

    def __call__(self, data: Data):
        idx = data.x_features_names.index("intensity")
        data.x[:, idx] = data.x[:, idx] / data.x[:, idx].max()
        idx = data.x_features_names.index("composite")
        data.x[:, idx] = self._standardize_channel(data.x[:, idx])
        return data

    def _standardize_channel(self, channel_data):
        """Sample-wise standardization y* = (y-y_mean)/y_std"""
        mean = channel_data.mean()
        std = channel_data.std()
        return (channel_data - mean) / std


class NormalizePos(BaseTransform):
    r"""
    Normalizes positions:
      - xy positions to be in the interval (-1, 1)
      - z position to start at 0.
      - preserve euclidian distances
    XYZ are expected to be centered already.
    """

    def __call__(self, data):
        xy_positive_amplitude = data.pos[:, :2].abs().max()
        xy_scale = (1 / xy_positive_amplitude) * 0.999999
        data.pos[:, :2] = data.pos[:, :2] * xy_scale
        data.pos[:, 2] = (data.pos[:, 2] - data.pos[:, 2].min()) * xy_scale

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class TargetTransform(BaseTransform):
    """
    Make target vector based on input classification dictionnary.

    Example:
    Source : y = [6,6,17,9,1]
    Pre-processed:
    - classification_preprocessing_dict = {17:1, 9:1}
    - y' = [6,6,1,1,1]
    Mapped to consecutive integers:
    - classification_dict = {1:"unclassified", 6:"building"}
    - y'' = [1,1,0,0,0]

    """

    def __init__(
        self,
        classification_preprocessing_dict: Dict[int, int],
        classification_dict: Dict[int, str],
    ):

        self._set_preprocessing_mapper(classification_preprocessing_dict)
        self._set_mapper(classification_dict)

    def __call__(self, data: Data):
        data.y = self.transform(data.y)
        return data

    def transform(self, y):
        y = self.preprocessing_mapper(y)
        y = self.mapper(y)
        return torch.LongTensor(y)

    def _set_preprocessing_mapper(self, classification_preprocessing_dict):
        """Set mapper from source classification code to another code."""
        d = {key: value for key, value in classification_preprocessing_dict.items()}
        self.preprocessing_mapper = np.vectorize(
            lambda class_code: d.get(class_code, class_code)
        )

    def _set_mapper(self, classification_dict):
        """Set mapper from source classification code to consecutive integers."""
        d = {
            class_code: class_index
            for class_index, class_code in enumerate(classification_dict.keys())
        }
        self.mapper = np.vectorize(lambda class_code: d.get(class_code))


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
    keys_to_long_format = ["pos", "x", "y", "pos_copy", "pos_copy_subsampled"]
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
                np.full(shape=len(data["pos_copy"]), fill_value=i)
                for i, data in enumerate(data_list)
            ]
        )
    )
    batch.batch_size = len(data_list)
    return batch


class ChannelNames(Enum):
    """Names of custom additional LAS channel"""

    PredictedClassification = "PredictedClassification"
    ProbasEntropy = "entropy"
