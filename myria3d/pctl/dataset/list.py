import copy
import glob
import os
from numbers import Number
from pathlib import Path
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
from myria3d.pctl.dataset.hdf5 import get_forest_classification_code

from myria3d.pctl.dataset.utils import (
    LAS_PATHS_BY_SPLIT_DICT_TYPE,
    SPLIT_TYPE,
    pdal_read_las_array_as_float32,
    pre_filter_below_n_points,
)
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform
from myria3d.utils import utils

log = utils.get_logger(__name__)


# def get_forest_classification_code(las_path):
#     species2code = {
#         "Abies_alba": 1,
#         "Abies_nordmanniana": 2,
#         "Castanea_sativa": 3,
#         "Fagus_sylvatica": 4,
#         "Larix_decidua": 5,
#         "Picea_abies": 6,
#         "Pinus_halepensis": 7,
#         "Pinus_nigra": 8,
#         "Pinus_nigra_laricio": 9,
#         "Pinus_pinaster": 10,
#         "Pinus_sylvestris": 11,
#         "Pseudotsuga_menziesii": 12,
#         "Quercus_ilex": 13,
#         "Quercus_petraea": 14,
#         "Quercus_pubescens": 15,
#         "Quercus_robur": 16,
#         "Quercus_rubra": 17,
#         "Robinia_pseudoacacia": 18,
#     }
#     for species in species2code:
#         if species in las_path:
#             return np.array([species2code[species]])
#     raise ValueError(las_path)


class ListDataset(Dataset):
    """Single-file HDF5 dataset for collections of large LAS tiles."""

    def __init__(
        self,
        las_paths_by_split_dict: LAS_PATHS_BY_SPLIT_DICT_TYPE,
        split,
        points_pre_transform: Callable = lidar_hd_pre_transform,
        subtile_overlap_train: Number = 0,
        pre_filter=pre_filter_below_n_points,
        train_transform: List[Callable] = None,
        eval_transform: List[Callable] = None,
    ):
        """Dataset for patches of lidar, already split together"""

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        self.subtile_overlap_train = subtile_overlap_train

        # Instantiates these to null;
        # They are loaded within __getitem__ to support multi-processing training.
        self.dataset = None

        # Use property once to be sure that samples are all indexed into the hdf5 file.
        self.samples_paths = las_paths_by_split_dict[split]

    def __getitem__(self, idx: int) -> Optional[Data]:
        sample_hdf5_path = self.samples_paths[idx]
        data = self._get_data(sample_hdf5_path)

        # filter if empty
        if data is None:
            return None

        if self.pre_filter and self.pre_filter(data):
            return None

        # Transforms, including sampling and some augmentations.
        transform = self.train_transform
        if sample_hdf5_path.startswith("val") or sample_hdf5_path.startswith("test"):
            transform = self.eval_transform
        if transform:
            data = transform(data)

        # filter if empty
        if not data or (self.pre_filter and self.pre_filter(data)):
            return None

        return data

    def _get_data(self, las_path: str) -> Data:
        points = pdal_read_las_array_as_float32(las_path)
        if not len(points):
            print(f"NO POINTS: {las_path}")
            return None
        data = self.points_pre_transform(points)
        data.x = torch.from_numpy(data.x)
        data.pos = torch.from_numpy(data.pos)
        data.y = torch.from_numpy(get_forest_classification_code(las_path))
        data.patch_id = Path(las_path).stem
        data.idx_in_original_cloud = np.arange(len(data.x))
        return data

    def __len__(self):
        return len(self.samples_paths)
