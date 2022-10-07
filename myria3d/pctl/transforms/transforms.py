import math
import re
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from myria3d.utils import utils

log = utils.get_logger(__name__)


class ToTensor(BaseTransform):
    """Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys: List[str] = ["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


class MinimumNumNodes(BaseTransform):
    def __init__(self, num: int):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes >= self.num:
            return data

        choice = torch.cat(
            [torch.randperm(num_nodes) for _ in range(math.ceil(self.num / num_nodes))],
            dim=0,
        )[: self.num]

        for key, item in data:
            if key == "num_nodes":
                data.num_nodes = choice.size(0)
            elif bool(re.search("edge", key)):
                continue
            elif (
                torch.is_tensor(item)
                and item.size(0) == num_nodes
                and item.size(0) != 1
            ):
                data[key] = item[choice]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num}"


class CopyFullPos:
    """Make a copy of the original positions - to be used for test and inference."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["pos_copy"] = data["pos"].clone()
        return data


class CopyFullPreparedTargets:
    """Make a copy of all, prepared targets - to be used for test."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["transformed_y_copy"] = data["y"].clone()
        return data


class CopySampledPos(BaseTransform):
    """Make a copy of the unormalized positions of subsampled points - to be used for test and inference."""

    def __call__(self, data: Data):
        if "copies" not in data:
            data.copies = dict()
        data.copies["pos_sampled_copy"] = data["pos"].clone()
        return data


class StandardizeRGBAndIntensity(BaseTransform):
    """Standardize RGB and log(Intensity) features."""

    def __call__(self, data: Data):
        idx = data.x_features_names.index("Intensity")
        # Log transform to be less sensitive to large outliers - info is in lower values
        data.x[:, idx] = torch.log(data.x[:, idx] + 1)
        data.x[:, idx] = self.standardize_channel(data.x[:, idx])
        idx = data.x_features_names.index("rgb_avg")
        data.x[:, idx] = self.standardize_channel(data.x[:, idx])
        return data

    def standardize_channel(self, channel_data: torch.Tensor, clamp_sigma: int = 3):
        """Sample-wise standardization y* = (y-y_mean)/y_std. clamping to ignore large values."""
        mean = channel_data.mean()
        std = channel_data.std() + 10**-6
        standard = (channel_data - mean) / std
        clamp = clamp_sigma * std
        clamped = torch.clamp(input=standard, min=-clamp, max=clamp)
        return clamped


class NullifyLowestZ(BaseTransform):
    """Center on x and y axis only. Set lowest z to 0."""

    def __call__(self, data):
        data.pos[:, 2] = data.pos[:, 2] - data.pos[:, 2].min()
        return data


class NormalizePos(BaseTransform):
    """
    Normalizes xy in [-1;1] range by scaling the whole point cloud (including z dim).
    XY are expected to be centered on zéro.

    """

    def __init__(self, subtile_width=50):
        half_subtile_width = subtile_width / 2
        self.scaling_factor = 1 / half_subtile_width

    def __call__(self, data):
        data.pos = data.pos * self.scaling_factor
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

        # Set to attribute to log potential type errors
        self.classification_dict = classification_dict
        self.classification_preprocessing_dict = classification_preprocessing_dict

    def __call__(self, data: Data):
        data.y = self.transform(data.y)
        return data

    def transform(self, y):
        y = self.preprocessing_mapper(y)
        try:
            y = self.mapper(y)
        except TypeError as e:
            log.error(
                "A TypeError occured when mapping target from arbitrary integers "
                "to consecutive integers (0-(n-1)) using the provided classification_dict "
                "This usually happens when an unknown classification code was encounterd. "
                "Check that all classification codes in your data are either "
                "specified via the classification_dict "
                "or transformed into a specified code via the preprocessing_mapper. \n"
                f"Current classification_dict: \n{self.classification_dict}\n"
                f"Current preprocessing_mapper: \n{self.classification_preprocessing_dict}\n"
                f"Current unique values in preprocessed target array: \n{np.unique(y)}\n"
            )
            raise e
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
