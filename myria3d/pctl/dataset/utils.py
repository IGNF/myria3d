import glob
import json
from pathlib import Path
import subprocess as sp
from numbers import Number
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
import pdal
from scipy.spatial import cKDTree

SPLIT_TYPE = Union[Literal["train"], Literal["val"], Literal["test"]]
LAS_PATHS_BY_SPLIT_DICT_TYPE = Dict[SPLIT_TYPE, List[str]]


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


def get_mosaic_of_centers(tile_width: Number, subtile_width: Number, subtile_overlap: Number = 0):
    if subtile_overlap < 0:
        raise ValueError("datamodule.subtile_overlap must be positive.")

    xy_range = np.arange(
        subtile_width / 2,
        tile_width + (subtile_width / 2) - subtile_overlap,
        step=subtile_width - subtile_overlap,
    )
    return [np.array([x, y]) for x in xy_range for y in xy_range]


def pdal_read_las_array(las_path: str, epsg: str):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path
        epsg (str): epsg to force the reading with

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.

    """
    p1 = pdal.Pipeline() | get_pdal_reader(las_path, epsg)
    p1.execute()
    return p1.arrays[0]


def pdal_read_las_array_as_float32(las_path: str, epsg: str):
    """Read LAS as a a named array, casted to floats."""
    arr = pdal_read_las_array(las_path, epsg)
    all_floats = np.dtype({"names": arr.dtype.names, "formats": ["f4"] * len(arr.dtype.names)})
    return arr.astype(all_floats)


def get_metadata(las_path: str) -> dict:
    """ returns metadata contained in a las file
    Args:
        las_path (str): input LAS path to get metadata from.
    Returns:
        dict : the metadata.
    """
    pipeline = pdal.Reader.las(filename=las_path).pipeline()
    pipeline.execute()
    return pipeline.metadata


def get_pdal_reader(las_path: str, epsg: str) -> pdal.Reader.las:
    """Standard Reader.
    Args:
        las_path (str): input LAS path to read.
        epsg (str): epsg to force the reading with
    Returns:
        pdal.Reader.las: reader to use in a pipeline.

    """

    if epsg :
        # if an epsg in provided, force pdal to read the lidar file with it
        return pdal.Reader.las(
            filename=las_path,
            nosrs=True,
            override_srs=f"EPSG:{epsg}",
        )

    # if 'srs' in get_metadata(las_path)['metadata']['readers.las'] and \
    #     'compoundwkt' in get_metadata(las_path)['metadata']['readers.las']['srs'] and \
    #     get_metadata(las_path)['metadata']['readers.las']['srs']['compoundwkt']:
    #     # read the lidar file with pdal default
    #     return pdal.Reader.las(filename=las_path)

    if 'srs' in get_metadata(las_path)['metadata']['readers.las']:
        if 'compoundwkt' in get_metadata(las_path)['metadata']['readers.las']['srs']:
            if get_metadata(las_path)['metadata']['readers.las']['srs']['compoundwkt']:
                # read the lidar file with pdal default
                return pdal.Reader.las(filename=las_path)

    raise Exception("No EPSG provided, neither in the lidar file or as parameter")


def get_pdal_info_metadata(las_path: str) -> Dict:
    """Read las metadata using pdal info
    Args:
        las_path (str): input LAS path to read.
    Returns:
        (dict): dictionary containing metadata from the las file
    """
    r = sp.run(["pdal", "info", "--metadata", las_path], capture_output=True)
    if r.returncode == 1:
        msg = r.stderr.decode()
        raise RuntimeError(msg)

    output = r.stdout.decode()
    json_info = json.loads(output)

    return json_info["metadata"]


# hdf5, iterable


def split_cloud_into_samples(
    las_path: str,
    tile_width: Number,
    subtile_width: Number,
    epsg: str,
    subtile_overlap: Number = 0,
):
    """Split LAS point cloud into samples.

    Args:
        las_path (str): path to raw LAS file
        tile_width (Number): width of input LAS file
        subtile_width (Number): width of receptive field.
        epsg (str): epsg to force the reading with
        subtile_overlap (Number, optional): overlap between adjacent tiles. Defaults to 0.

    Yields:
        _type_: idx_in_original_cloud, and points of sample in pdal input format casted as floats.

    """
    points = pdal_read_las_array_as_float32(las_path, epsg)
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    kd_tree = cKDTree(pos[:, :2] - pos[:, :2].min(axis=0))
    XYs = get_mosaic_of_centers(tile_width, subtile_width, subtile_overlap=subtile_overlap)
    for center in XYs:
        radius = subtile_width // 2  # Square receptive field.
        minkowski_p = np.inf
        sample_idx = np.array(kd_tree.query_ball_point(center, r=radius, p=minkowski_p))
        if not len(sample_idx):
            # no points in this receptive fields
            continue
        sample_points = points[sample_idx]
        yield sample_idx, sample_points


def pre_filter_below_n_points(data, min_num_nodes=1):
    return data.pos.shape[0] < min_num_nodes


def get_las_paths_by_split_dict(
    data_dir: str, split_csv_path: str
) -> LAS_PATHS_BY_SPLIT_DICT_TYPE:
    las_paths_by_split_dict: LAS_PATHS_BY_SPLIT_DICT_TYPE = {}
    split_df = pd.read_csv(split_csv_path)
    for phase in ["train", "val", "test"]:
        basenames = split_df[split_df.split == phase].basename.tolist()
        # Reminder: an explicit data structure with ./val, ./train, ./test subfolder is required.
        las_paths_by_split_dict[phase] = [str(Path(data_dir) / phase / b) for b in basenames]

    if not las_paths_by_split_dict:
        raise FileNotFoundError(
            (
                f"No basename found while parsing directory {data_dir}"
                f"using {split_csv_path} as split CSV."
            )
        )

    return las_paths_by_split_dict
