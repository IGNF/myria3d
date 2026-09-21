"""Generation of a toy dataset for testing purposes."""

import os
import os.path as osp
import sys

from torch_geometric.transforms import GridSampling

# to use from CLI.
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.dirname(__file__)))))
from myria3d.pctl.dataset.hdf5 import HDF5Dataset  # noqa
from myria3d.pctl.transforms.compose import CustomCompose  # noqa
from myria3d.pctl.transforms.transforms import (  # noqa
    DropPointsByClass,
    TargetTransform,
)

TOY_EPSG = "2154"
TOY_LAS_DATA = "tests/data/toy_dataset_src/862000_6652000.classified_toy_dataset.100mx100m.las"
TOY_DATASET_HDF5_PATH = "tests/data/toy_dataset.hdf5"

# Deterministic transforms baked into the HDF5 file once, at dataset-creation time,
# for the TRAIN split only (applied in create_hdf5). They are therefore NOT recomputed
# every epoch, which is a large speed-up on big datasets. Here, we mirror them in the toy dataset creation, so that the toy dataset is consistent with the default config's `train_bake` list.
# creation.

TOY_CLASSIFICATION_PREPROCESSING_DICT = {3: 5, 4: 5, 0: 1, 66: 65, 67: 1, 100: 1, 101: 1}

TOY_CLASSIFICATION_DICT = {
    1: "unclassified",
    2: "ground",
    5: "vegetation",
    6: "building",
    9: "water",
    17: "bridge",
    64: "lasting_above",
}


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

    # Deterministic transforms baked into the train split, mirroring the default config's
    # `train_bake` list. TargetTransform is required so that stored train targets are mapped
    # to consecutive integers (0-(n-1)); this mapping is no longer applied at load time.
    train_pre_transform = CustomCompose(
        [
            TargetTransform(
                TOY_CLASSIFICATION_PREPROCESSING_DICT,
                TOY_CLASSIFICATION_DICT,
            ),
            DropPointsByClass(),
            GridSampling(0.25),
        ]
    )

    HDF5Dataset(
        TOY_DATASET_HDF5_PATH,
        TOY_EPSG,
        las_paths_by_split_dict={
            "train": [TOY_LAS_DATA],
            "val": [TOY_LAS_DATA],
            "test": [TOY_LAS_DATA],
        },
        tile_width=110,
        subtile_width=50,
        train_pre_transform=train_pre_transform,
        train_transform=None,
        eval_transform=None,
        pre_filter=None,
    )
    return TOY_DATASET_HDF5_PATH


if __name__ == "__main__":
    make_toy_dataset_from_test_file()
