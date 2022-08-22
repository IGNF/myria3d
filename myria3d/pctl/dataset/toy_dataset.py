"""Generation of a toy dataset for testing purposes."""

import os
import os.path as osp
import sys

# to use from CLI.
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))
from myria3d.pctl.dataset.hdf5 import HDF5Dataset  # noqa


TOY_LAS_DATA = "tests/data/toy_dataset_src/870000_6618000.subset.50mx100m.las"
TOY_SPLIT_CSV = "tests/data/toy_dataset_src/toy_dataset_split.csv"
TOY_DATASET_HDF5_PATH = "tests/data/toy_dataset.hdf5"


def make_toy_dataset_from_test_file():
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
    if os.path.isfile(TOY_DATASET_HDF5_PATH):
        os.remove(TOY_DATASET_HDF5_PATH)

    # TODO: update transforms ? or use a config ?
    HDF5Dataset(
        TOY_DATASET_HDF5_PATH,
        las_files_by_split={
            "train": [TOY_LAS_DATA],
            "val": [TOY_LAS_DATA],
            "test": [TOY_LAS_DATA],
        },
        tile_width=110,
        subtile_width=50,
        train_transform=None,
        eval_transform=None,
        pre_filter=None,
    )
    return TOY_DATASET_HDF5_PATH


if __name__ == "__main__":
    make_toy_dataset_from_test_file()
