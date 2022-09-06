import glob
import math
from numbers import Number
from typing import Dict, List, Literal, Union
import pdal
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
SHAPE_TYPE = Union[Literal["disk"], Literal["square"]]
LAS_FILES_BY_SPLIT_TYPE = Dict[SPLIT_TYPE, List[str]]

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
        sample_idx = np.array(kd_tree.query_ball_point(center, r=radius, p=minkowski_p))
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
