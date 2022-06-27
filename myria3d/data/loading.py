"""How to represent points clouds, specifying specific data signatures (e.g. choice of features).

Country-specific definitions inherit from abstract class LidarDataLogic, which implements general logics
to split each point cloud into subtiles, format these subtiles, and save a learning-ready dataset splitted into
train, val. A test set is also created, which is simply a copy of the selected test point clouds. This is so
test phase conditions are similar to "in-the-wild" prediction conditions.

In particular, subclasses implement a "load_las" method that is used by the datamodule at test and inference time.

"""

import argparse
import copy
import math
import os
import glob
import os.path as osp
from shutil import copyfile
import shutil
from tqdm import tqdm
import laspy
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from scipy.spatial import cKDTree

SPLIT_PHASES = ["val", "train", "test"]
# Min num of nodes per sub-cloud to be considered
MIN_NUM_NODES_PER_RECEPTIVE_FIELD = 50

# Files for the creation of a toy dataset.
LAS_SUBSET_FOR_TOY_DATASET = (
    "tests/data/toy_dataset_src/870000_6618000.subset.50mx100m.las"
)
SPLIT_CSV_FOR_TOY_DATASET = "tests/data/toy_dataset_src/toy_dataset_split.csv"


class LidarDataSignature:
    """Abstract class to load, chunk, and save a point cloud dataset according to a train/val/test split.
    load_las and its needed parameters ares specified in child classes.

    """

    return_number_normalization_max_value = 7

    def __init__(
        self,
        input_data_dir: str = None,
        prepared_data_dir: str = None,
        split_csv: str = None,
        input_tile_width_meters: int = 1000,
        subtile_width_meters: int = 50,
        **kwargs,
    ):
        self.input_data_dir = input_data_dir
        self.prepared_data_dir = prepared_data_dir
        self.split_csv = split_csv
        self.input_tile_width_meters = input_tile_width_meters
        self.subtile_width_meters = subtile_width_meters
        # radius of a circle that contains the subtile of square shape subtile_width_meters * subtile_width_meters
        # The aera of this circle is (pi/2) = 1.5708 times the area of the original square.
        self.sample_radius = subtile_width_meters / math.sqrt(2)

    def prepare_full_dataset(self):
        """Prepare a dataset for model training and model evaluation.

        Iterates through LAS files listed in a csv metadata file.
        A `split` column specifies the train/val/test split of the dataset to be created.

        """
        split_df = pd.read_csv(self.split_csv)
        for phase in tqdm(SPLIT_PHASES, desc="Phases"):
            basenames = split_df[split_df.split == phase].basename.tolist()
            print(f"Subset: {phase}")
            print("  -  ".join(basenames))
            for file_basename in tqdm(basenames, desc="Files"):
                las_path = self._find_file_in_dir(self.input_data_dir, file_basename)
                if phase == "test":
                    # Simply copy the LAS to the new test folder.
                    test_subdir_path = osp.join(self.prepared_data_dir, phase)
                    os.makedirs(test_subdir_path, exist_ok=True)
                    copy_path = osp.join(test_subdir_path, file_basename)
                    copyfile(las_path, copy_path)
                elif phase in ["train", "val"]:
                    # Load LAS into memory as a Data object with selected features,
                    # then iteratively extract 50m*50m subtiles by filtering along x
                    # then y axis. Serialize the resulting Data object using torch.save.
                    subdir_path = osp.join(
                        self.prepared_data_dir, phase, osp.basename(las_path)
                    )
                    os.makedirs(subdir_path, exist_ok=True)
                    self.split_and_save(las_path, subdir_path)
                else:
                    raise KeyError("Phase should be one of train/val/test.")

    def split_and_save(self, filepath: str, output_subdir_path: str) -> None:
        for idx, data_sample in enumerate(
            self.split_cloud_into_receptive_fields(filepath)
        ):
            out_path = osp.join(output_subdir_path, f"{str(idx).zfill(4)}.data")
            torch.save(data_sample, out_path)

    def split_cloud_into_receptive_fields(self, filepath: str):
        """Iterator that parses a LAS, extract receptive fields as Data objects.

        Args:
            filepath (str): input LAS file
            output_subdir_path (str): output directory to save splitted `.data` objects.

        """
        data = self.load_las(filepath)
        # KDTree for fast query based on x and y that starts at 0
        kd_tree = cKDTree(data.pos[:, :2] - data.pos[:, :2].min(axis=0))

        range_by_axis = np.arange(
            self.subtile_width_meters / 2,
            self.input_tile_width_meters + self.subtile_width_meters / 2,
            self.subtile_width_meters,
        )

        for x_center in range_by_axis:
            for y_center in range_by_axis:
                center = np.array([x_center, y_center])
                data_sample = self.extract_around_center(data, kd_tree, center)
                if (
                    data_sample
                    and len(data_sample.pos) >= MIN_NUM_NODES_PER_RECEPTIVE_FIELD
                ):
                    yield data_sample

    def extract_around_center(
        self, data: Data, kd_tree: cKDTree, center: np.array
    ) -> Data:
        """Filter a data object on a chosen axis, using a relative position .
        Modifies the original data object so that extracted future filters are faster.

        Args:
            data (Data): a pyg Data object with pos, x, and y attributes.
            relative_pos (int): where the data to extract start on chosen axis (typically in range 0-1000)
            axis (int, optional): 0 for x and 1 for y axis. Defaults to 0.

        Returns:
            Data: the data that is at most subtile_width_meters above relative_pos on the chosen axis.

        """
        # query
        circular_sample_idx = np.array(
            kd_tree.query_ball_point(center, r=self.sample_radius)
        )

        if len(circular_sample_idx) == 0:
            return None

        sample_data = Data()
        sample_data.x_features_names = copy.deepcopy(data.x_features_names)
        sample_data.las_filepath = copy.deepcopy(data.las_filepath)
        sample_data.pos = data.pos[circular_sample_idx]
        sample_data.x = data.x[circular_sample_idx]
        sample_data.y = data.y[circular_sample_idx]
        sample_data.idx_in_original_cloud = data.idx_in_original_cloud[
            circular_sample_idx
        ]

        return sample_data

    def _find_file_in_dir(self, input_data_dir: str, basename: str) -> str:
        """Query files matching a basename in input_data_dir and its subdirectories.

        Args:
            input_data_dir (str): data directory

        Returns:
            [str]: first file path matching the query.

        """
        query = f"{input_data_dir}/**/{basename}"
        files = glob.glob(query, recursive=True)
        return files[0]


class FrenchLidarDataSignature(LidarDataSignature):

    x_features_names = [
        "intensity",
        "return_number",
        "number_of_returns",
        "red",
        "green",
        "blue",
        "nir",
        "rgb_avg",
        "ndvi",
    ]
    colors_normalization_max_value = 255 * 256

    def load_las(cls, las_filepath: str):
        f"""Loads a point cloud in LAS format to memory and turns it into torch-geometric Data object.

        Builds a composite (average) color channel on the fly.

        Calculate NDVI on the fly.

        x_features_names are {' - '.join(cls.x_features_names)}

        Args:
            las_filepath (str): path to the LAS file.

        Returns:
            Data: the point cloud formatted for later deep learning training.

        """
        las = laspy.read(las_filepath)

        # Positions and base features
        pos = np.asarray([las.x, las.y, las.z], dtype=np.float32).transpose()
        x = np.asarray(
            [
                las[x_name]
                for x_name in [
                    "intensity",
                    "return_number",
                    "number_of_returns",
                    "red",
                    "green",
                    "blue",
                    "nir",
                ]
            ],
            dtype=np.float32,
        ).transpose()

        # normalization
        return_number_idx = cls.x_features_names.index("return_number")
        occluded_points = x[:, return_number_idx] > 1

        x[:, return_number_idx] = (x[:, return_number_idx]) / (
            cls.return_number_normalization_max_value
        )
        num_return_idx = cls.x_features_names.index("number_of_returns")
        x[:, num_return_idx] = (x[:, num_return_idx]) / (
            cls.return_number_normalization_max_value
        )

        for idx, c in enumerate(cls.x_features_names):
            if c in ["red", "green", "blue", "nir"]:
                print(
                    x[:, idx].max()
                )  # DEBUG: just to be sure that it is the same as before
                assert x[:, idx].max() <= cls.colors_normalization_max_value
                x[:, idx] = x[:, idx] / cls.colors_normalization_max_value
                x[occluded_points, idx] = 0

        red = x[:, cls.x_features_names.index("red")]
        green = x[:, cls.x_features_names.index("green")]
        blue = x[:, cls.x_features_names.index("blue")]

        # Additional features :
        # Average color, that will be normalized on the fly based on single-sample
        rgb_avg = np.asarray([red, green, blue], dtype=np.float32).mean(axis=0)

        # NDVI
        nir = x[:, cls.x_features_names.index("nir")]
        ndvi = (nir - red) / (nir + red + 10**-6)
        x = np.concatenate([x, rgb_avg[:, None], ndvi[:, None]], axis=1)

        try:
            # for LAS format V1.2
            y = las.classification.array.astype(int)
        except Exception:
            # for  LAS format V1.4
            y = las.classification.astype(int)

        return Data(
            pos=pos,
            x=x,
            y=y,
            las_filepath=las_filepath,
            x_features_names=cls.x_features_names,
            idx_in_original_cloud=np.arange(len(pos)),
        )

    def per_sample_standardization(self, data: Data):
        """Scale features in 0-1 range.

        Additionnaly : use reserved -0.75 value for occluded points colors(normal range is -0.5 to 0.5).

        """

        idx = data.x_features_names.index("intensity")
        # Log transform to be less sensitive to large outliers - info is in lower values
        data.x[:, idx] = torch.log(data.x[:, idx] + 1)
        data.x[:, idx] = standardize_channel(data.x[:, idx])
        idx = data.x_features_names.index("rgb_avg")
        data.x[:, idx] = standardize_channel(data.x[:, idx])

        return data


class SwissTopoLidarDataLogic(LidarDataSignature):
    x_features_names = [
        "intensity",
        "return_number",
        "number_of_returns",
        "red",
        "green",
        "blue",
        "rgb_avg",
    ]
    colors_normalization_max_value = 256

    @classmethod
    def load_las(cls, las_filepath: str) -> Data:
        """Loads a point cloud in LAS format to memory and turns it into torch-geometric Data object.

        Builds a composite (average) color channel on the fly.

        Args:
            las_filepath (str): path to the LAS file.

        Returns:
            Data: the point cloud formatted for later deep learning training.

        """
        las = laspy.read(las_filepath)
        pos = np.asarray([las.x, las.y, las.z], dtype=np.float32).transpose()

        x = np.asarray(
            [
                las[x_name]
                for x_name in [
                    "intensity",
                    "return_number",
                    "number_of_returns",
                    "red",
                    "green",
                    "blue",
                ]
            ],
            dtype=np.float32,
        ).transpose()

        return_number_idx = cls.x_features_names.index("return_number")
        occluded_points = x[:, return_number_idx] > 1

        x[:, return_number_idx] = (x[:, return_number_idx]) / (
            cls.return_number_normalization_max_value
        )
        num_return_idx = cls.x_features_names.index("number_of_returns")
        x[:, num_return_idx] = (x[:, num_return_idx]) / (
            cls.return_number_normalization_max_value
        )

        for idx, c in enumerate(cls.x_features_names):
            if c in ["red", "green", "blue"]:
                assert x[:, idx].max() <= cls.colors_normalization_max_value
                x[:, idx] = x[:, idx] / cls.colors_normalization_max_value
                x[occluded_points, idx] = 0

        rgb_avg = (
            np.asarray(
                [las[x_name] for x_name in ["red", "green", "blue"]], dtype=np.float32
            )
            .transpose()
            .mean(axis=1, keepdims=True)
        )

        x = np.concatenate([x, rgb_avg], axis=1)

        try:
            # for LAS format V1.2
            y = las.classification.array.astype(int)
        except Exception:
            # for  LAS format V1.4
            y = las.classification.astype(int)

        return Data(
            pos=pos,
            x=x,
            y=y,
            las_filepath=las_filepath,
            x_features_names=cls.x_features_names,
            idx_in_original_cloud=np.arange(len(pos)),
        )

    def per_sample_standardization(self, data: Data):
        """Scale features in 0-1 range.

        Additionnaly : use reserved -0.75 value for occluded points colors(normal range is -0.5 to 0.5).

        """

        idx = data.x_features_names.index("intensity")
        # Log transform to be less sensitive to large outliers - info is in lower values
        data.x[:, idx] = torch.log(data.x[:, idx] + 1)
        data.x[:, idx] = standardize_channel(data.x[:, idx])
        idx = data.x_features_names.index("rgb_avg")
        data.x[:, idx] = standardize_channel(data.x[:, idx])

        return data


def standardize_channel(channel_data: torch.Tensor, clamp_sigma: int = 3):
    """Sample-wise standardization y* = (y-y_mean)/y_std"""
    mean = channel_data.mean()
    std = channel_data.std() + 10**-6
    standard = (channel_data - mean) / std
    clamp = clamp_sigma * std
    clamped = torch.clamp(input=standard, min=-clamp, max=clamp)
    return clamped


# Scripts for data preparation and preparation of a toy dataset for tests


def make_toy_dataset_from_test_file(
    prepared_data_dir: str,
    src_las_path: str = LAS_SUBSET_FOR_TOY_DATASET,
    split_csv: str = SPLIT_CSV_FOR_TOY_DATASET,
):
    """Prepare a toy dataset from a single, small LAS file.

    The file is first duplicated to get 2 LAS in each split (train/val/test),
    and then each file is splitted into .data files, resulting in a training-ready
    dataset loacted in td_prepared

    Args:
        src_las_path (str): input, small LAS file to generate toy dataset from
        split_csv (str): Path to csv with a `basename` (e.g. '123_456.las') and
        a `split` (train/val/test) columns specifying the dataset split.
        prepared_data_dir (str): where to copy files (`raw` subfolder) and to prepare
        dataset files (`prepared` subfolder)

    Returns:
        str: path to directory containing prepared dataset.

    """
    # Copy input file for full test isolation
    td_raw = osp.join(prepared_data_dir, "raw")
    td_prepared = osp.join(prepared_data_dir, "prepared")
    os.makedirs(td_raw)
    os.makedirs(td_prepared)
    # Make a "raw", unporcessed dataset with six files.
    basename = osp.basename(src_las_path)
    for s in ["train1", "train2", "val1", "val2", "test1", "test2"]:
        copy_path = osp.join(td_raw, basename.replace(".las", f".{s}.las"))
        shutil.copy(src_las_path, copy_path)

    # Prepare a Deep-Learning-ready dataset, using the split defined in the csv.
    data_prepper = FrenchLidarDataSignature(
        input_data_dir=td_raw,
        prepared_data_dir=td_prepared,
        split_csv=split_csv,
        input_tile_width_meters=110,
        subtile_width_meters=50,
    )
    data_prepper.prepare_full_dataset()
    return td_prepared


def _get_data_preparation_parser():
    """Gets a parser with parameters for dataset preparation.

    Returns:
        argparse.ArgumentParser: the parser.
    """
    parser = argparse.ArgumentParser(
        description="Prepare a Lidar dataset for deep learning.",
        epilog="If you need a toy dataset, you only need to"
        " set --origin=FR_TOY and specify a value for --prepared_data_dir.",
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        default="./data/raw/",
        help="Path to directory with las files with unique basename and `.las` suffix.",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        default="./split.csv",
        help="Path to csv with a basename (e.g. '123_456.las') and split (train/val/test) columns specifying the dataset split.",
    )
    parser.add_argument(
        "--prepared_data_dir",
        type=str,
        default="./data/prepared/",
        help="Path to folder to save Data object train/val/test subfolders.",
    )
    parser.add_argument(
        "--origin", type=str, default="FR", choices=["FR", "CH", "FR_TOY"]
    )

    return parser


def main():
    """Code to execute to prepare a new set of Lidar tiles for training 3D segmentaiton neural networks.

    From a data directory containing point cloud in LAS format, and a scv specifying the dataset
    train/val/test split for each file (columns: split, basename, example: "val","123_456.las"),
    split the dataset, chunk the point cloud tiles into smaller subtiles, and prepare each sample
    as a pytorch geometric Data object.

    Echo numbers and colors are scaled to be in 0-1 range. Intensity and average color
    are not scaled b/c they are expected to be standardized later.

    To show help relative to data preparation, run:

        cd myria3d/datamodules/

        conda activate myria3d

        python loading.py -h


    """

    parser = _get_data_preparation_parser()
    args = parser.parse_args()
    if args.origin == "FR_TOY":
        make_toy_dataset_from_test_file(args.prepared_data_dir)
    elif args.origin == "FR":
        data_prepper = FrenchLidarDataSignature(**args.__dict__)
        data_prepper.prepare_full_dataset()
    elif args.origin == "CH":
        data_prepper = SwissTopoLidarDataLogic(**args.__dict__)
        data_prepper.prepare_full_dataset()
    else:
        raise KeyError(f"Data origin is invalid (currently: {args.origin})")


if __name__ == "__main__":
    main()
