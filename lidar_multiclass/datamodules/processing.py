# pylint: disable
import copy
import os
import os.path as osp
import math
from enum import Enum
from typing import Callable, Dict, List, AnyStr

import laspy
import numpy as np
import torch
from torch_geometric.nn.pool import knn
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

from lidar_multiclass.utils import utils

log = utils.get_logger(__name__)

# CONSTANTS
UNIT = 1
HALF_UNIT = 0.5


def load_las_data(
    data_filepath,
    features_names=[
        "intensity",
        "return_num",
        "num_returns",
        "red",
        "green",
        "blue",
        "composite",
    ],
):
    """
    Load a cloud of points and its labels. LAS Format: 1.2.
    Shape: [n_points, n_features].
    Warning: las.x is in meters, las.X is in centimeters.
    """

    log.debug(f"Loading {data_filepath}")
    las = laspy.read(data_filepath)

    features_names = copy.deepcopy(features_names)
    las_features_name = [f for f in features_names if f not in ["composite"]]

    pos = np.asarray(
        [
            las.x,
            las.y,
            las.z,
        ],
        dtype=np.float32,
    ).transpose()
    x = np.asarray(
        [las[x_name] for x_name in las_features_name],
        dtype=np.float32,
    ).transpose()

    colors = ["red", "green", "blue"]
    commposite = (
        np.asarray(
            [las[x_name] for x_name in colors],
            dtype=np.float32,
        )
        .transpose()
        .mean(axis=1, keepdims=True)
    )
    x = np.concatenate([x, commposite], axis=1)

    # TODO: assure that post-preparation data are always in LAS Format1.4
    try:
        # LAS format V1.2
        y = las.classification.array.astype(np.int)
    except:
        # LAS format V1.4
        y = las.classification.astype(np.int)

    full_cloud_filepath = get_full_las_filepath(data_filepath)

    return Data(
        pos=pos,
        x=x,
        y=y,
        data_filepath=data_filepath,
        full_cloud_filepath=full_cloud_filepath,
        x_features_names=features_names,
    )


def get_full_las_filepath(data_filepath):
    """
    Return the reference to the full LAS file from which data_flepath depends.
    Predict mode: return data_filepath
    Train/val/test mode: lasfile_dir/test/tile_id/tile_id_SUB1234.las -> /lasfile_dir/colorized/tile_id.las
    """
    # Predict mode: we use the full tile path as id directly without processing
    if "_SUB" not in data_filepath:
        return data_filepath

    # Else, we need the path of the full las to save predictions in test/eval.
    basename = osp.basename(data_filepath)
    stem = basename.split("_SUB")[0]
    stem = stem.replace("-", "_")

    lasfile_dir = osp.dirname(osp.dirname(osp.dirname(osp.dirname(data_filepath))))
    return osp.join(lasfile_dir, "colorized", stem + ".las")


# DATA TRANSFORMS


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


class SelectPredictSubTile:
    r"""
    Select a specified square subtile from a tile to infer on.
    Returns None if there are no candidate building points.
    """

    def __init__(
        self,
        subtile_width_meters: float = 50.0,
    ):
        self.subtile_width_meters = subtile_width_meters

    def __call__(self, data: Data):

        subtile_data = self.get_subtile_data(data, data.current_subtile_center)
        if len(subtile_data.pos) > 0:
            return subtile_data
        return None

    def get_subtile_data(self, data: Data, subtile_center_xy):
        """Extract tile points and labels around a subtile center using Chebyshev distance, in meters."""
        subtile_data = data.clone()

        chebyshev_distance = np.max(
            np.abs(subtile_data.pos[:, :2] - subtile_center_xy), axis=1
        )
        mask = chebyshev_distance <= (self.subtile_width_meters / 2)

        subtile_data.pos = subtile_data.pos[mask]
        subtile_data.x = subtile_data.x[mask]
        subtile_data.y = subtile_data.y[mask]

        return subtile_data


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


class CustomNormalizeFeatures(BaseTransform):
    r"""
    Scale features in 0-1 range.
    Additionnaly : use reserved -0.75 value for occluded points colors(normal range is -0.5 to 0.5).
    """

    def __init__(
        self,
        colors_normalization_max_value: int,
        return_num_normalization_max_value: int,
    ):
        self.standard_colors_names = ["red", "green", "blue", "nir", "composite"]
        self.colors_normalization_max_value = colors_normalization_max_value
        self.return_num_normalization_max_value = return_num_normalization_max_value

    def __call__(self, data: Data):

        intensity_idx = data.x_features_names.index("intensity")
        data.x[:, intensity_idx] = (
            data.x[:, intensity_idx] / data.x[:, intensity_idx].max() - HALF_UNIT
        )

        return_num_idx = data.x_features_names.index("return_num")
        colors_idx = []
        for color_name in self.standard_colors_names:
            if color_name in data.x_features_names:
                colors_idx.append(data.x_features_names.index(color_name))

        for color_idx in colors_idx:
            data.x[:, color_idx] = (
                data.x[:, color_idx] / self.colors_normalization_max_value - HALF_UNIT
            )
            data.x[data.x[:, return_num_idx] > 1, color_idx] = -1.5 * HALF_UNIT

        composite_idx = data.x_features_names.index("composite")
        clamp_value = -3
        data.x[:, composite_idx] = self._standardize_channel(
            data.x[:, composite_idx]
        ).clamp(min=-5)
        data.x[data.x[:, return_num_idx] > 1, composite_idx] = 1.5 * clamp_value

        data.x[:, return_num_idx] = (data.x[:, return_num_idx] - UNIT) / (
            self.return_num_normalization_max_value - UNIT
        ) - HALF_UNIT
        num_return_idx = data.x_features_names.index("num_returns")
        data.x[:, num_return_idx] = (data.x[:, num_return_idx] - UNIT) / (
            self.return_num_normalization_max_value - UNIT
        ) - HALF_UNIT

        return data

    def _standardize_channel(self, channel_data):
        """Sample-wise standardization y* = (y-y_mean)/y_std"""
        mean = channel_data.mean()
        std = channel_data.std()
        return (channel_data - mean) / std


class CustomNormalizeScale(BaseTransform):
    r"""
    Normalizes node positions to the interval (-1, 1).
    XYZ are expected to be centered already. Normalization is performed
    by a single xy positive amplitude to preserve euclidian distances.
    Typically, xy_positive_amplitude = width / 2
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
    """

    def __init__(self, classification_dict: List[AnyStr]):
        self.classification_mapper = {
            class_code: class_index
            for class_index, class_code in enumerate(classification_dict.keys())
        }

    def __call__(
        self,
        data: Data,
    ):
        data.y = self.transform(data.y)
        return data

    def transform(self, y):
        y = np.vectorize(self.classification_mapper.get)(y)
        y = torch.LongTensor(y)
        return y


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


class DataHandler:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        output_dir: str,
        classification_dict: Dict[int, str],
        names_of_probas_to_save: List[str] = [],
    ):

        os.makedirs(output_dir, exist_ok=True)
        self.preds_dirpath = output_dir
        self.current_full_cloud_filepath = ""
        self.classification_dict = classification_dict
        self.names_of_probas_to_save = names_of_probas_to_save

        self.reverse_classification_mapper = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }

        self.index_of_probas_to_save = [
            list(classification_dict.values()).index(name)
            for name in names_of_probas_to_save
        ]

    @torch.no_grad()
    def update_with_inference_outputs(self, outputs: dict, prefix: str = ""):
        """
        Save the predicted classes in las format with position.
        Handle las loading when necessary.

        :param outputs: outputs of a step.
        :param prefix: train, val, test or predict (str).
        """
        batch = outputs["batch"].detach()
        batch_classification = outputs["preds"].detach()
        batch_probas = outputs["proba"][:, self.index_of_probas_to_save].detach()
        if self.index_of_probas_to_save:
            pass
        for batch_idx, full_cloud_filepath in enumerate(batch.full_cloud_filepath):
            is_a_new_tile = full_cloud_filepath != self.current_full_cloud_filepath
            if is_a_new_tile:
                close_previous_las_first = self.current_full_cloud_filepath != ""
                if close_previous_las_first:
                    self.interpolate_and_save(prefix)
                self._load_las_for_classification_update(full_cloud_filepath)

            idx_x = batch.batch_x == batch_idx
            # TODO: add probas if needed
            self.updates_classification_subsampled.append(batch_classification[idx_x])
            self.updates_probas_subsampled.append(batch_probas[idx_x])
            self.updates_pos_subsampled.append(batch.pos_copy_subsampled[idx_x])
            idx_y = batch.batch_y == batch_idx
            self.updates_pos.append(batch.pos_copy[idx_y])

    def _load_las_for_classification_update(self, filepath):
        """Load a LAS and add necessary extradim."""

        self.las = laspy.read(filepath)
        self.current_full_cloud_filepath = filepath

        coln = ChannelNames.PredictedClassification.value
        param = laspy.ExtraBytesParams(name=coln, type=int)
        self.las.add_extra_dim(param)
        self.las[coln][:] = 0

        for class_name in self.names_of_probas_to_save:
            param = laspy.ExtraBytesParams(name=class_name, type=float)
            self.las.add_extra_dim(param)
            self.las[class_name][:] = 0.0

        self.las_pos = torch.from_numpy(
            np.asarray(
                [
                    self.las.x,
                    self.las.y,
                    self.las.z,
                ],
                dtype=np.float32,
            ).transpose()
        )
        self.updates_classification_subsampled = []
        self.updates_probas_subsampled = []
        self.updates_pos_subsampled = []
        self.updates_pos = []

    @torch.no_grad()
    def interpolate_and_save(self, prefix):
        """
        Interpolate all predicted probabilites to their original points in LAS file, and save.
        Returns the path of the updated, saved LAS file.
        """

        basename = os.path.basename(self.current_full_cloud_filepath)
        if prefix:
            basename = f"{prefix}_{basename}"

        os.makedirs(self.preds_dirpath, exist_ok=True)
        self.output_path = os.path.join(self.preds_dirpath, basename)

        # Cat
        self.updates_pos = torch.cat(self.updates_pos).cpu()
        self.updates_pos_subsampled = torch.cat(self.updates_pos_subsampled).cpu()
        self.updates_probas_subsampled = torch.cat(self.updates_probas_subsampled).cpu()
        self.updates_classification_subsampled = torch.cat(
            self.updates_classification_subsampled
        ).cpu()

        # Remap predictions
        self.updates_classification_subsampled = np.vectorize(
            self.reverse_classification_mapper.get
        )(self.updates_classification_subsampled)
        self.updates_classification_subsampled = torch.from_numpy(
            self.updates_classification_subsampled
        )

        # 1/2 Interpolate locally to have dense classes in infered zones
        assign_idx = knn(
            self.updates_pos_subsampled,
            self.updates_pos,
            k=1,
            num_workers=1,
        )[1]
        self.updates_classification = self.updates_classification_subsampled[assign_idx]
        self.updates_probas = self.updates_probas_subsampled[assign_idx]

        # 2/2 Propagate dense classes to the full las
        assign_idx = knn(
            self.las_pos,
            self.updates_pos,
            k=1,
            num_workers=1,
        )[1]
        assign_idx = assign_idx
        self.las[ChannelNames.PredictedClassification.value][
            assign_idx
        ] = self.updates_classification

        for class_idx_in_tensor, class_name in enumerate(self.names_of_probas_to_save):
            self.las[class_name][assign_idx] = self.updates_probas[
                :, class_idx_in_tensor
            ]

        log.info(f"Saving LAS updated with predicted classes to {self.output_path}")
        self.las.write(self.output_path)

        # Clean-up - get rid of current data to go easy on memory
        self.current_full_cloud_filepath = ""
        del self.las
        del self.las_pos
        del self.updates_pos_subsampled
        del self.updates_pos
        del self.updates_classification_subsampled
        del self.updates_classification
        del self.updates_probas_subsampled

        return self.output_path
