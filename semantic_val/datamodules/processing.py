# pylint: disable
import os
import os.path as osp
import math
from enum import Enum
from typing import Callable, List

import laspy
import numpy as np
import torch
from torch_geometric.nn.pool import knn
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform
from semantic_val.decision.codes import MTS_AUTO_DETECTED_CODE, reset_classification

from semantic_val.utils import utils

log = utils.get_logger(__name__)

# CONSTANTS

# Warning: be sure that this order matches the one in load_las_data.
COLORS_NAMES = ["red", "green", "blue", "nir"]
X_FEATURES_NAMES = [
    "intensity",
    "return_num",
    "num_returns",
] + COLORS_NAMES

INTENSITY_MAX = 32768.0
COLORS_MAX = 255 * 256
MAX_TRY_IN_TRAIN_MODE = 25
RETURN_NUM_MAX = 7

HALF_UNIT = 0.5
UNIT = 1


class ChannelNames(Enum):
    """Names of custom and standard LAS channel"""

    # Standard
    Classification = "classification"
    ClusterID = "ClusterID"

    # Custom
    BDTopoOverlay = "BDTopoOverlay"
    BuildingsProba = "BuildingsProba"


def get_full_las_filepath(data_filepath):
    """
    Return the reference to the full LAS file from which data_flepath depends.
    Predict mode: return data_filepath
    Train/val/test mode: lasfile_dir/split/tile_id/tile_id_SUB1234.las -> /lasfile_dir/tile_id.las
    """
    # Predict mode: we use the full tile path as id directly without processing
    if "/split/" not in data_filepath:
        return data_filepath

    # Else, we need the path of the full las to save predictions in test/eval.
    basename = osp.basename(data_filepath)
    stem = basename.split("_SUB")[0]
    # /

    lasfile_dir = osp.dirname(osp.dirname(osp.dirname(data_filepath)))
    return osp.join(lasfile_dir, stem + ".las")


def load_las_data(data_filepath):
    """
    Load a cloud of points and its labels.
    Shape: [n_points, n_features].
    Warning: las.x is in meters, las.X is in centimeters.
    """
    log.debug(f"Loading {data_filepath}")
    las = laspy.read(data_filepath)

    # ANALYSIS: UNCOMMENT TO PREDICT ON FILE THAT WENT THROUGH INSPECTION
    las.classification = reset_classification(las.classification)

    pos = np.asarray(
        [
            las.x,
            las.y,
            las.z,
        ],
        dtype=np.float32,
    ).transpose()
    x = np.asarray(
        [las[x_name] for x_name in X_FEATURES_NAMES],
        dtype=np.float32,
    ).transpose()
    y = las.classification.astype(np.int)

    full_cloud_filepath = get_full_las_filepath(data_filepath)

    return Data(
        pos=pos,
        x=x,
        y=y,
        data_filepath=data_filepath,
        full_cloud_filepath=full_cloud_filepath,
        x_features_names=X_FEATURES_NAMES,
    )


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
        mts_auto_detected_code: int = MTS_AUTO_DETECTED_CODE,
    ):
        self.subtile_width_meters = subtile_width_meters
        self.mts_auto_detected_code = mts_auto_detected_code

    def __call__(self, data: Data):

        subtile_data = self.get_subtile_data(data, data.current_subtile_center)
        if len(subtile_data.pos) > 0:
            num_candidate_building_points = np.sum(
                subtile_data.y == self.mts_auto_detected_code
            )
            if num_candidate_building_points > 0:
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
    r"""Make a copy of the full cloud's positions and labels, for inference interpolation."""

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


# TODO: OK - Until we change and do not need to keep them for train ; will still be necessary for test
# if we use the full las.
class MakeCopyOfPosAndY(BaseTransform):
    r"""Make a copy of the full cloud's positions and labels, for inference interpolation."""

    def __call__(self, data: Data):
        data["pos_copy"] = data["pos"].clone()
        data["y_copy"] = data["y"].clone()
        return data


# TODO: OK
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


# TODO: OK
class MakeCopyOfSampledPos(BaseTransform):
    """Make a copy of the unormalized positions of subsampled points."""

    def __call__(self, data: Data):
        data["pos_copy_subsampled"] = data["pos"].clone()
        return data


class RandomTranslateFeatures(BaseTransform):
    r"""
    Randomly translate the (unnormalized) features values.

    Intensity: random translate by rel_translation * max
    Colors (RGB): random translate by rel_translation * max
    Number of returns: +1/+0/-1 with equal probability
    Return number: +1/+0/-1 with equal probability, max-clamped by number of returns.
    """

    def __call__(self, data: Data, rel_translation: float = 0.02):

        x = data.x
        (n, _) = x.size()

        translation = rel_translation * INTENSITY_MAX
        intensity_idx = data.x_features_names.index("intensity")
        delta = x[:, intensity_idx].new_empty(n).uniform_(-translation, translation)
        x[:, intensity_idx] = x[:, intensity_idx] + delta
        x[:, intensity_idx] = x[:, intensity_idx].clamp(min=0, max=INTENSITY_MAX)

        translation = rel_translation * COLORS_MAX
        COLORS_IDX = [
            data.x_features_names.index(color_name) for color_name in COLORS_NAMES
        ]
        for color_idx in COLORS_IDX:
            delta = x[:, color_idx].new_empty(n).uniform_(-translation, translation)
            x[:, color_idx] = x[:, color_idx] + delta
            x[:, color_idx] = x[:, color_idx].clamp(min=0, max=COLORS_MAX)

        num_return_idx = data.x_features_names.index("num_returns")
        delta = x[:, num_return_idx].new_empty(n).random_(-1, 2)
        x[:, num_return_idx] = x[:, num_return_idx] + delta
        x[:, num_return_idx] = x[:, num_return_idx].clamp(min=1, max=RETURN_NUM_MAX)

        return_num_idx = data.x_features_names.index("return_num")
        delta = x[:, return_num_idx].new_empty(n).random_(-1, 2)
        x[:, return_num_idx] = x[:, return_num_idx] + delta
        x[:, return_num_idx] = x[:, return_num_idx].clamp(min=1)
        x[:, return_num_idx] = torch.min(x[:, return_num_idx], x[:, num_return_idx])

        return data


class CustomNormalizeFeatures(BaseTransform):
    r"""Scale features in 0-1 range."""

    def __call__(self, data: Data):

        intensity_idx = data.x_features_names.index("intensity")
        data.x[:, intensity_idx] = data.x[:, intensity_idx] / INTENSITY_MAX - HALF_UNIT

        colors_idx = [
            data.x_features_names.index(color_name) for color_name in COLORS_NAMES
        ]
        for color_idx in colors_idx:
            data.x[:, color_idx] = data.x[:, color_idx] / COLORS_MAX - HALF_UNIT

        return_num_idx = data.x_features_names.index("return_num")
        data.x[:, return_num_idx] = (data.x[:, return_num_idx] - UNIT) / (
            RETURN_NUM_MAX - UNIT
        ) - HALF_UNIT
        num_return_idx = data.x_features_names.index("num_returns")
        data.x[:, num_return_idx] = (data.x[:, num_return_idx] - UNIT) / (
            RETURN_NUM_MAX - UNIT
        ) - HALF_UNIT

        return data


class CustomNormalizeScale(BaseTransform):
    r"""Normalizes node positions to the interval (-1, 1)."""

    def __init__(self, z_scale: float = 100.0):
        self.z_scale = z_scale
        pass

    def __call__(self, data):

        xy_scale = (1 / data.pos[:, :2].abs().max()) * 0.999999
        data.pos[:, :2] = data.pos[:, :2] * xy_scale

        data.pos[:, 2] = (data.pos[:, 2] - data.pos[:, 2].min()) / self.z_scale

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class MakeBuildingTargets(BaseTransform):
    """
    Pass from multiple classes to simpler Building/Non-Building labels.
    Initial classes: [  1,   2,   6 (detected building, no validation),  19 (valid building),  20 (surdetection, unspecified),
    21 (building, forgotten), 104, 110 (surdetection, others), 112 (surdetection, vehicule), 114 (surdetection, others), 115 (surdetection, bridges)]
    Final classes: 0 (non-building), 1 (building)
    Applied on both unsampled and subsampled labels (only because target_transforms are called after transforms)
    """

    def __call__(self, data: Data, keys: List[str] = ["y", "y_copy"]):
        for key in keys:
            data[key] = self.make_building_targets(data[key])
        return data

    def make_building_targets(self, y):
        buildings_idx = (y == 19) | (y == 21)
        y[buildings_idx] = 1
        y[~buildings_idx] = 0
        return y


# TODO: we may not need all attributes. To be updated.
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
    keys_to_long_format = ["pos", "x", "y", "pos_copy", "y_copy", "pos_copy_subsampled"]
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
                np.full(shape=len(data["y_copy"]), fill_value=i)
                for i, data in enumerate(data_list)
            ]
        )
    )
    batch.batch_size = len(data_list)
    return batch


class DataHandler:
    """A class to load, update with proba, and save a LAS."""

    def __init__(self, preds_dirpath: str = ""):

        os.makedirs(preds_dirpath, exist_ok=True)
        self.preds_dirpath = preds_dirpath
        self.current_full_cloud_filepath = ""

    def load_las_for_proba_update(self, filepath):
        """Load a LAS and add necessary extradims."""

        self.las = laspy.read(filepath)
        self.current_full_cloud_filepath = filepath
        # ANALYSIS: UNCOMMENT TO PREDICT ON FILE THAT WENT THROUGH INSPECTION
        self.las.classification = reset_classification(self.las.classification)

        coln = ChannelNames.BuildingsProba.value
        param = laspy.ExtraBytesParams(name=coln, type=float)
        self.las.add_extra_dim(param)
        self.las[coln][:] = 0.0

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
        # "Never incrementally cat results; append to a list instead"
        self.proba_updates_subsampled = []
        self.pos_updates_subsampled = []
        self.pos_updates = []

    @torch.no_grad()
    def append_pos_and_proba_to_list(self, outputs: dict, phase: str = ""):
        """
        Save the predicted classes in las format with position. Load the las if necessary.

        :param outputs: outputs of a step.
        :param phase: train, val or test phase (str).
        """
        batch = outputs["batch"].detach()
        batch_proba = outputs["proba"].detach()

        for batch_idx, full_cloud_filepath in enumerate(batch.full_cloud_filepath):
            is_a_new_tile = self.current_full_cloud_filepath != full_cloud_filepath
            if is_a_new_tile:
                close_previous_las_first = self.current_full_cloud_filepath != ""
                if close_previous_las_first:
                    self.interpolate_probas_and_save(phase)
                self.load_las_for_proba_update(full_cloud_filepath)

            idx_x = batch.batch_x == batch_idx
            self.proba_updates_subsampled.append(batch_proba[idx_x, 1])
            self.pos_updates_subsampled.append(batch.pos_copy_subsampled[idx_x])
            idx_y = batch.batch_y == batch_idx
            self.pos_updates.append(batch.pos_copy[idx_y])

    @torch.no_grad()
    def interpolate_probas_and_save(self, phase):
        """
        Interpolate all predicted probabilites to their original points in LAS file, and save.
        Returns the path of the updated LAS file.
        """

        tile_basename = os.path.basename(self.current_full_cloud_filepath)
        filename = f"{phase}_{tile_basename}"
        os.makedirs(self.preds_dirpath, exist_ok=True)
        self.output_path = os.path.join(self.preds_dirpath, filename)

        self.proba_updates_subsampled = torch.cat(self.proba_updates_subsampled)
        self.pos_updates_subsampled = torch.cat(self.pos_updates_subsampled)
        self.pos_updates = torch.cat(self.pos_updates)
        self.las_pos = self.las_pos.to(self.pos_updates_subsampled.device)

        # IDEA: KNN interpolate has high memory cost.
        # An alternative would be knn with k=3 and taking the average of the pk nearest proba

        # 1/2 Interpolate locally to have dense probas in infered zones
        assign_idx = knn(
            self.pos_updates_subsampled,
            self.pos_updates,
            k=1,
            num_workers=1,
        )[1]
        self.proba_updates = self.proba_updates_subsampled[assign_idx]

        # 2/2 Propagate dense probas to the full las
        assign_idx = knn(
            self.las_pos,
            self.pos_updates,
            k=1,
            num_workers=1,
        )[1]
        self.las[ChannelNames.BuildingsProba.value][
            assign_idx
        ] = self.proba_updates.cpu()
        log.info(f"Saving LAS updated with predicted probas to {self.output_path}")
        self.las.write(self.output_path)

        # Clean-up - get rid of current data to go easy on memory
        self.current_full_cloud_filepath = ""
        del self.las
        del self.las_pos
        del self.pos_updates_subsampled
        del self.pos_updates
        del self.proba_updates_subsampled
        del self.proba_updates

        return self.output_path
