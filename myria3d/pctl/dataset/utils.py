import os
import glob
import math
import copy
from numbers import Number
from typing import Callable, Dict, List, Literal, Union, Optional
import h5py
import pdal
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm
from torch_geometric.data import Data
import pandas as pd
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform


SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
SHAPE_TYPE = Union[Literal["disk"], Literal["square"]]
LAS_PATHS_BY_SPLIT_DICT_TYPE = Dict[SPLIT_TYPE, List[str]]

# commons


def find_file_in_dir(data_dir: str, basename: str) -> str:
    """Query files matching a basename in input_data_dir and its subdirectories.
    Args:
        input_data_dir (str): data directory
    Returns:
        [str]: first file path matching the query.
    """
    query = f"{data_dir}/**/{basename}"
    files = glob.glob(query, recursive=True)
    return files[0]


def get_mosaic_of_centers(
    tile_width: Number, subtile_width: Number, subtile_overlap: Number = 0
):
    if subtile_overlap < 0:
        raise ValueError("datamodule.subtile_overlap must be positive.")

    xy_range = np.arange(
        subtile_width / 2,
        tile_width + (subtile_width / 2) - subtile_overlap,
        step=subtile_width - subtile_overlap,
    )
    return [np.array([x, y]) for x in xy_range for y in xy_range]


def pdal_read_las_array(las_path: str):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.

    """
    p1 = pdal.Pipeline() | pdal.Reader.las(filename=las_path)
    p1.execute()
    return p1.arrays[0]


def pdal_read_las_array_as_float32(las_path: str):
    """Read LAS as a a named array, casted to floats."""
    arr = pdal_read_las_array(las_path)
    all_floats = np.dtype(
        {"names": arr.dtype.names, "formats": ["f4"] * len(arr.dtype.names)}
    )
    return arr.astype(all_floats)


def get_pdal_reader(las_path: str) -> pdal.Reader.las:
    """Standard Reader which imposes Lamber 93 SRS.
    Args:
        las_path (str): input LAS path to read.
    Returns:
        pdal.Reader.las: reader to use in a pipeline.

    """
    return pdal.Reader.las(
        filename=las_path,
        nosrs=True,
        override_srs="EPSG:2154",
    )


# hdf5, iterable


def split_cloud_into_samples(
    las_path: str,
    tile_width: Number,
    subtile_width: Number,
    shape: SHAPE_TYPE,
    subtile_overlap: Number = 0,
):
    """Split LAS point cloud into samples.

    Args:
        las_path (str): path to raw LAS file
        tile_width (Number): width of input LAS file
        subtile_width (Number): width of receptive field ; may be increased for coverage in case of disk shape.
        shape: "disk" or "square"
        subtile_overlap (Number, optional): overlap between adjacent tiles. Defaults to 0.

    Yields:
        _type_: idx_in_original_cloud, and points of sample in pdal input format casted as floats.

    """
    points = pdal_read_las_array_as_float32(las_path)
    pos = np.asarray(
        [points["X"], points["Y"], points["Z"]], dtype=np.float32
    ).transpose()
    kd_tree = cKDTree(pos[:, :2] - pos[:, :2].min(axis=0))
    XYs = get_mosaic_of_centers(
        tile_width, subtile_width, subtile_overlap=subtile_overlap
    )
    for center in tqdm(XYs, desc="Centers"):
        radius = subtile_width // 2  # Square receptive field.
        minkowski_p = np.inf
        if shape == "disk":
            # Disk receptive field.
            # Adapt radius to have complete coverage of the data, with a slight overlap between samples.
            minkowski_p = 2
            radius = radius * math.sqrt(2)
        sample_idx = np.array(
            kd_tree.query_ball_point(center, r=radius, p=minkowski_p)
        )
        if not len(sample_idx):
            # no points in this receptive fields
            continue
        sample_points = points[sample_idx]
        yield sample_idx, sample_points


def pre_filter_below_n_points(data, min_num_nodes=50):
    return data.pos.shape[0] < min_num_nodes


# COPC


def get_random_center_in_tile(tile_width, subtile_width):
    return np.random.randint(
        subtile_width / 4,
        tile_width - (subtile_width / 4) + 1,
        size=(2,),
    )


def make_circle_wkt(center, subtile_width):
    half = subtile_width / 2
    wkt = Point(center).buffer(half).wkt
    return wkt


def get_las_paths_by_split_dict(data_dir: str, split_csv_path: str) -> LAS_PATHS_BY_SPLIT_DICT_TYPE:
    las_paths_by_split_dict: LAS_PATHS_BY_SPLIT_DICT_TYPE = {}
    split_df = pd.read_csv(split_csv_path)
    for phase in ["train", "val", "test"]:
        basenames = split_df[
            split_df.split == phase
        ].basename.tolist()
        las_paths_by_split_dict[phase] = [
            find_file_in_dir(data_dir, b) for b in basenames
        ]

    if not las_paths_by_split_dict:
        raise FileNotFoundError(
            (
                f"No basename found while parsing directory {data_dir}"
                f"using {split_csv_path} as split CSV."
            )
        )

    return las_paths_by_split_dict


def create_hdf5(
    las_paths_by_split_dict: dict,
    hdf5_file_path: str,
    tile_width: Number = 1000,
    subtile_width: Number = 50,
    subtile_shape: SHAPE_TYPE = "square",
    pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
    subtile_overlap_train: Number = 0,
    points_pre_transform: Callable = lidar_hd_pre_transform
):

    """Create a HDF5 dataset file from las.
    Args:
        split (str): specifies either "train", "val", or "test" split.
        las_path (str): path to point cloud.

    """
    os.makedirs(os.path.dirname(hdf5_file_path), exist_ok=True)
    for split, las_paths in las_paths_by_split_dict.items():
        with h5py.File(hdf5_file_path, "a") as f:
            if split not in f:
                f.create_group(split)
        for las_path in tqdm(las_paths, desc=f"Preparing {split} set..."):

            basename = os.path.basename(las_path)

            # Delete dataset for incomplete LAS entry, to start from scratch.
            # Useful in case data preparation was interrupted.
            with h5py.File(hdf5_file_path, "a") as hdf5_file:
                if basename in hdf5_file[split] and "is_complete" not in hdf5_file[split][basename].attrs:
                    del hdf5_file[basename]
                    # Parse and add subtiles to split group.
            with h5py.File(hdf5_file_path, "a") as hdf5_file:
                if basename in hdf5_file[split]:
                    continue

                subtile_overlap = subtile_overlap_train if split == "train" else 0  # No overlap at eval time.
                for sample_number, (sample_idx, sample_points) in enumerate(split_cloud_into_samples(
                    las_path,
                    tile_width,
                    subtile_width,
                    subtile_shape,
                    subtile_overlap,
                )):
                    if not points_pre_transform:
                        continue
                    data = points_pre_transform(sample_points)
                    if pre_filter is not None and pre_filter(data):
                        # e.g. pre_filter spots situations where num_nodes is too small.
                        continue
                    hdf5_path = os.path.join(split, basename, str(sample_number).zfill(5))
                    hd5f_path_x = os.path.join(hdf5_path, "x")
                    hdf5_file.create_dataset(
                        hd5f_path_x,
                        data.x.shape,
                        dtype="f",
                        data=data.x,
                    )
                    hdf5_file[hd5f_path_x].attrs["x_features_names"] = copy.deepcopy(
                        data.x_features_names
                    )
                    hdf5_file.create_dataset(
                        os.path.join(hdf5_path, "pos"),
                        data.pos.shape,
                        dtype="f",
                        data=data.pos,
                    )
                    hdf5_file.create_dataset(
                        os.path.join(hdf5_path, "y"),
                        data.y.shape,
                        dtype="i",
                        data=data.y,
                    )
                    hdf5_file.create_dataset(
                        os.path.join(hdf5_path, "idx_in_original_cloud"),
                        sample_idx.shape,
                        dtype="i",
                        data=sample_idx,
                    )

                # A termination flag to report that all samples for this point cloud were included in the df5 file.
                hdf5_file[split][basename].attrs["is_complete"] = True
