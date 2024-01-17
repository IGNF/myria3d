import math
import re
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from myria3d.utils import utils

log = utils.get_logger(__name__)

COMMON_CODE_FOR_ALL_ARTEFACTS = 65


class ToTensor(BaseTransform):
    """Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys: List[str] = ["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


def subsample_data(data, num_nodes, choice):
    # TODO: get num_nodes from data.num_nodes instead to simplify signature
    for key, item in data:
        if key == "num_nodes":
            data.num_nodes = choice.size(0)
        elif bool(re.search("edge", key)):
            continue
        elif torch.is_tensor(item) and item.size(0) == num_nodes and item.size(0) != 1:
            data[key] = item[choice]
    return data


class MaximumNumNodes(BaseTransform):
    def __init__(self, num: int):
        self.num = num

    def __call__(self, data):
        num_nodes = data.num_nodes

        if num_nodes <= self.num:
            return data

        choice = torch.randperm(data.num_nodes)[: self.num]
        data = subsample_data(data, num_nodes, choice)

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

        data = subsample_data(data, num_nodes, choice)

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
        if torch.isnan(std):
            std = 1.0
        standard = (channel_data - mean) / std
        clamp = clamp_sigma * std
        clamped = torch.clamp(input=standard, min=-clamp, max=clamp)
        return clamped


class AblationOfAllColors(BaseTransform):
    """Nullify all colors information."""

    def __call__(self, data: Data):
        for f in ["Red", "Green", "Blue", "Infrared", "rgb_avg", "ndvi"]:
            idx = data.x_features_names.index(f)
            data.x[:, idx] = 0.0
        return data


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


class RandomTranslate(BaseTransform):
    """
    shift randomly x and y
    """

    def __init__(self, max_random_shift):
        self.max_random_shift = max_random_shift

    def __call__(self, data):
        scale_x = random.uniform(-self.max_random_shift, self.max_random_shift)
        scale_y = random.uniform(-self.max_random_shift, self.max_random_shift)
        data.pos[:, 0] = data.pos[:, 0] + scale_x
        data.pos[:, 1] = data.pos[:, 1] + scale_y
        return data


# ugly hgack to deal with hydra quickly...
class RandomScale(BaseTransform):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    (functional name: :obj:`random_scale`).

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scales: Tuple[float, float]) -> None:
        scales = [scales[0], scales[1]]
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data: Data) -> Data:
        assert data.pos is not None

        scale = random.uniform(*self.scales)
        data.pos = data.pos * scale
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.scales})"


def split_idx_by_dim(dim_array):
    """
    Returns a sequence of arrays of indices of elements sharing the same value in dim_array
    Groups are ordered by ascending value.
    """
    idx = np.argsort(dim_array)
    sorted_dim_array = dim_array[idx]
    group_idx = np.array_split(idx, np.where(np.diff(sorted_dim_array) != 0)[0] + 1)
    return group_idx


def cart2pol(x, y):
    rho = torch.sqrt(x**2 + y**2)
    phi = torch.arctan2(y, x)
    return (rho, phi)


NUM_TREE_STATISTICS = 8
NUM_TREE_FEATURES = 4


class TreeStatistics(BaseTransform):
    """
    Computes tree-level statistics.
    """

    def __call__(self, data):
        # We first prepare the data by removing ClusterID==0 -> the non-tree elements
        data = subsample_data(data, len(data.x), data.cluster_id != 0)

        # Then we handcraft tree features
        tree_statistics = torch.zeros((len(data.x), NUM_TREE_STATISTICS))
        tree_features = torch.zeros((len(data.x), NUM_TREE_FEATURES))

        for tree_idx in split_idx_by_dim(data.cluster_id):
            tree_pos = data.pos[tree_idx]
            x = tree_pos[:, 0]
            y = tree_pos[:, 1]
            z = tree_pos[:, 2]

            # tree stats
            height = torch.max(z) - torch.min(z)

            # position within tree
            relative_z = (z - torch.min(z)) / height
            relative_z_mean = torch.mean(relative_z)
            relative_z_std = torch.std(relative_z)

            # polar coordinates
            rho, phi = cart2pol(x - torch.mean(x), y - torch.mean(y))
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            # more stats
            rho_max = torch.max(rho)
            relative_rho = rho / rho_max
            relative_rho_mean = torch.mean(relative_rho)
            relative_rho_std = torch.std(relative_rho)
            approx_area = rho_max**2
            approx_volume = height * approx_area

            stats = (
                torch.stack(
                    [
                        height,
                        relative_z_mean,
                        relative_z_std,
                        rho_max,
                        relative_rho_mean,
                        relative_rho_std,
                        approx_area,
                        approx_volume,
                    ]
                )
                .repeat(len(x))
                .view(len(x), -1)
            )
            tree_statistics[tree_idx] = stats

            features = torch.stack(
                [
                    relative_z,
                    relative_rho,
                    cos_phi,
                    sin_phi,
                ]
            ).transpose(0, 1)
            tree_features[tree_idx] = features
        data.tree_statistics = tree_statistics
        data.tree_features = tree_features
        return data


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
        self.preprocessing_mapper = np.vectorize(lambda class_code: d.get(class_code, class_code))

    def _set_mapper(self, classification_dict):
        """Set mapper from source classification code to consecutive integers."""
        d = {
            class_code: class_index
            for class_index, class_code in enumerate(classification_dict.keys())
        }
        # Here we update the dict so that code 65 remains unchanged.
        # Indeed, 65 is reserved for noise/artefacts points, that will be deleted by transform "DropPointsByClass".
        d.update({65: 65})
        self.mapper = np.vectorize(lambda class_code: d.get(class_code))


class DropPointsByClass(BaseTransform):
    """Drop points with class -1 (i.e. artefacts that would have been mapped to code -1)"""

    def __call__(self, data):
        points_to_drop = torch.isin(data.y, COMMON_CODE_FOR_ALL_ARTEFACTS)
        if points_to_drop.sum() > 0:
            points_to_keep = torch.logical_not(points_to_drop)
            data = subsample_data(data, num_nodes=data.num_nodes, choice=points_to_keep)
            # Here we also subsample these idx since we do not need to interpolate these points back
            if "idx_in_original_cloud" in data:
                data.idx_in_original_cloud = data.idx_in_original_cloud[points_to_keep]
        return data
