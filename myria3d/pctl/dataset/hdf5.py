import copy
import os
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from myria3d.pctl.dataset.utils import (
    LAS_PATHS_BY_SPLIT_DICT_TYPE,
    SPLIT_TYPE,
    pre_filter_below_n_points,
    split_cloud_into_samples,
)
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform
from myria3d.utils import utils

log = utils.get_logger(__name__)


def get_forest_classification_code(las_path):
    species2code = {
        "Abies_alba": 1,
        "Abies_nordmanniana": 2,
        "Castanea_sativa": 3,
        "Fagus_sylvatica": 4,
        "Larix_decidua": 5,
        "Picea_abies": 6,
        "Pinus_halepensis": 7,
        "Pinus_nigra": 8,
        "Pinus_nigra_laricio": 9,
        "Pinus_pinaster": 10,
        "Pinus_sylvestris": 11,
        "Pseudotsuga_menziesii": 12,
        "Quercus_ilex": 13,
        "Quercus_petraea": 14,
        "Quercus_pubescens": 15,
        "Quercus_robur": 16,
        "Quercus_rubra": 17,
        "Robinia_pseudoacacia": 18,
    }
    for species in species2code:
        if species in las_path:
            return np.array([species2code[species]])
    raise ValueError(las_path)


class HDF5Dataset(Dataset):
    """Single-file HDF5 dataset for collections of large LAS tiles."""

    def __init__(
        self,
        hdf5_file_path: str,
        las_paths_by_split_dict: LAS_PATHS_BY_SPLIT_DICT_TYPE,
        points_pre_transform: Callable = lidar_hd_pre_transform,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap_train: Number = 0,
        pre_filter=pre_filter_below_n_points,
        train_transform: List[Callable] = None,
        eval_transform: List[Callable] = None,
    ):
        """Initialization, taking care of HDF5 dataset preparation if needed, and indexation of its content.

        Args:
            las_paths_by_split_dict (Optional[LAS_PATHS_BY_SPLIT_DICT_TYPE]): should look like
                las_paths_by_split_dict = {'train': ['dir/las1.las','dir/las2.las'], 'val': [...], , 'test': [...]}
            hdf5_file_path (str): path to HDF5 dataset
            points_pre_transform (Callable): Function to turn pdal points into a pyg Data object.
            tile_width (Number, optional): width of a LAS tile. Defaults to 1000.
            subtile_width (Number, optional): effective width of a subtile (i.e. receptive field). Defaults to 50.
            subtile_overlap_train (Number, optional): Overlap for data augmentation of train set. Defaults to 0.
            pre_filter (_type_, optional): Function to filter out specific subtiles. Defaults to None.
            train_transform (List[Callable], optional): Transforms to apply to a sample for training. Defaults to None.
            eval_transform (List[Callable], optional): Transforms to apply to a sample for evaluation (test/val sets). Defaults to None.

        """

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap_train = subtile_overlap_train

        self.hdf5_file_path = hdf5_file_path

        # Instantiates these to null;
        # They are loaded within __getitem__ to support multi-processing training.
        self.dataset = None
        self._samples_hdf5_paths = None

        if not las_paths_by_split_dict:
            log.warning(
                "No las_paths_by_split_dict given, pre-computed HDF5 dataset is therefore used."
            )
            return

        # Add data for all LAS Files into a single hdf5 file.
        create_hdf5(
            las_paths_by_split_dict,
            hdf5_file_path,
            tile_width,
            subtile_width,
            pre_filter,
            subtile_overlap_train,
            points_pre_transform,
        )

        # Use property once to be sure that samples are all indexed into the hdf5 file.
        self.samples_hdf5_paths

    def __getitem__(self, idx: int) -> Optional[Data]:
        sample_hdf5_path = self.samples_hdf5_paths[idx]
        data = self._get_data(sample_hdf5_path)

        # filter if empty
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

    def _get_data(self, sample_hdf5_path: str) -> Data:
        """Loads a Data object from the HDF5 dataset.

        Opening the file has a high cost so we do it only once and store the opened files as a singleton
        for each process within __get_item__ and not in __init__ to support for Multi-GPU.

        See https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16?u=piojanu.

        """
        if self.dataset is None:
            self.dataset = h5py.File(self.hdf5_file_path, "r")

        grp = self.dataset[sample_hdf5_path]
        # [...] needed to make a copy of content and avoid closing HDF5.
        # Nota: idx_in_original_cloud SHOULD be np.ndarray, in order to be batched into a list,
        # which serves to keep track of indivual sample sizes in a simpler way for interpolation.
        return Data(
            x=torch.from_numpy(grp["x"][...]),
            pos=torch.from_numpy(grp["pos"][...]),
            y=torch.from_numpy(grp["y"][...]),
            idx_in_original_cloud=grp["idx_in_original_cloud"][...],
            x_features_names=grp["x"].attrs["x_features_names"].tolist(),
            # num_nodes=grp["pos"][...].shape[0],  # Not needed - performed under the hood.
        )

    def __len__(self):
        return len(self.samples_hdf5_paths)

    @property
    def traindata(self):
        return self._get_split_subset("train")

    @property
    def valdata(self):
        return self._get_split_subset("val")

    @property
    def testdata(self):
        return self._get_split_subset("test")

    def _get_split_subset(self, split: SPLIT_TYPE):
        """Get a sub-dataset of a specific (train/val/test) split."""
        indices = [idx for idx, p in enumerate(self.samples_hdf5_paths) if p.startswith(split)]
        return torch.utils.data.Subset(self, indices)

    @property
    def samples_hdf5_paths(self):
        """Index all samples in the dataset, if not already done before."""
        # Use existing if already loaded as variable.
        if self._samples_hdf5_paths:
            return self._samples_hdf5_paths

        # Load as variable if already indexed in hdf5 file. Need to decode b-string.
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            if "samples_hdf5_paths" in hdf5_file:
                self._samples_hdf5_paths = [
                    sample_path.decode("utf-8") for sample_path in hdf5_file["samples_hdf5_paths"]
                ]
                return self._samples_hdf5_paths

        # Otherwise, index samples, and add the index as an attribute to the HDF5 file.
        self._samples_hdf5_paths = []
        with h5py.File(self.hdf5_file_path, "r") as hdf5_file:
            for split in hdf5_file.keys():
                if split not in ["train", "val", "test"]:
                    continue
                for basename in hdf5_file[split].keys():
                    for sample_number in hdf5_file[split][basename].keys():
                        self._samples_hdf5_paths.append(osp.join(split, basename, sample_number))

        with h5py.File(self.hdf5_file_path, "a") as hdf5_file:
            # special type to avoid silent string truncation in hdf5 datasets.
            variable_lenght_str_datatype = h5py.special_dtype(vlen=str)
            hdf5_file.create_dataset(
                "samples_hdf5_paths",
                (len(self.samples_hdf5_paths),),
                dtype=variable_lenght_str_datatype,
                data=self._samples_hdf5_paths,
            )
        return self._samples_hdf5_paths


def create_hdf5(
    las_paths_by_split_dict: dict,
    hdf5_file_path: str,
    tile_width: Number = 1000,
    subtile_width: Number = 50,
    pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
    subtile_overlap_train: Number = 0,
    points_pre_transform: Callable = lidar_hd_pre_transform,
):
    """Create a HDF5 dataset file from las.

    Args:
        split (str): specifies either "train", "val", or "test" split.
        las_path (str): path to point cloud.

        las_paths_by_split_dict ([LAS_PATHS_BY_SPLIT_DICT_TYPE]): should look like
                las_paths_by_split_dict = {'train': ['dir/las1.las','dir/las2.las'], 'val': [...], , 'test': [...]},
        hdf5_file_path (str): path to HDF5 dataset,
        tile_width (Number, optional): width of a LAS tile. 1000 by default,
        subtile_width: (Number, optional): effective width of a subtile (i.e. receptive field). 50 by default,
        pre_filter: Function to filter out specific subtiles. "pre_filter_below_n_points" by default,
        subtile_overlap_train (Number, optional): Overlap for data augmentation of train set. 0 by default,
        points_pre_transform (Callable): Function to turn pdal points into a pyg Data object.

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
                if (
                    basename in hdf5_file[split]
                    and "is_complete" not in hdf5_file[split][basename].attrs
                ):
                    del hdf5_file[split][basename]
                    # Parse and add subtiles to split group.
            with h5py.File(hdf5_file_path, "a") as hdf5_file:
                if basename in hdf5_file[split]:
                    continue

                subtile_overlap = (
                    subtile_overlap_train if split == "train" else 0
                )  # No overlap at eval time.
                for sample_number, (sample_idx, sample_points) in enumerate(
                    split_cloud_into_samples(
                        las_path,
                        tile_width,
                        subtile_width,
                        subtile_overlap,
                    )
                ):
                    if not points_pre_transform:
                        continue
                    data = points_pre_transform(sample_points)
                    data.y = get_forest_classification_code(las_path)
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
                # Group may not have been created if source cloud had no patch passing the pre_filter step, hence the "if" here.
                if basename in hdf5_file[split]:
                    hdf5_file[split][basename].attrs["is_complete"] = True
