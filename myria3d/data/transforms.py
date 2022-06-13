from typing import Callable, Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from myria3d.data.loading import LidarDataSignature
from myria3d.utils import utils

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


class ToTensor(BaseTransform):
    """Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys=["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


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


class NormalizedFolowingDataLogic(BaseTransform):
    """Apply the normalization specified in the data_signature"""

    def __init__(self, data_signature: LidarDataSignature):
        self.data_signature = data_signature

    def __call__(self, data: Data):
        data = self.data_signature.per_sample_standardization(data)
        return data


class NullifyLowestZ(BaseTransform):
    """Center on x and y axis only. Set lowest z to 0."""

    def __call__(self, data):
        data.pos[:, 2] = data.pos[:, 2] - data.pos[:, 2].min()
        return data


class NormalizePos(BaseTransform):
    """
    Normalizes xy in [-1;1] range by scaling the whole point cloud (including z dim)
    XY are expected to be centered on zéro.

    """

    def __init__(self, subtile_width_meters=50):
        half_subtile_width_meters = subtile_width_meters / 2
        self.scaling_factor = 1 / half_subtile_width_meters

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
