import copy
import os
import os.path as osp
from numbers import Number
from typing import Callable, List, Optional
import h5py

import torch
from torch_geometric.data import Data
from tqdm import tqdm
from torch.utils.data import Dataset
from myria3d.pctl.points_pre_transform.lidar_hd import lidar_hd_pre_transform

from myria3d.pctl.dataset.utils import (
    SHAPE_TYPE,
    SPLIT_TYPE,
    LAS_FILES_BY_SPLIT_TYPE,
    pre_filter_below_n_points,
    split_cloud_into_samples,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)


class HDF5Dataset(Dataset):
    """Single-file HDF5 dataset for collections of large LAS tiles."""

    def __init__(
        self,
        hdf5_file_path: str,
        las_files_by_split: Optional[LAS_FILES_BY_SPLIT_TYPE] = None,
        points_pre_transform: Callable = lidar_hd_pre_transform,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap_train: Number = 0,
        subtile_shape: SHAPE_TYPE = "square",
        pre_filter=pre_filter_below_n_points,
        train_transform: List[Callable] = None,
        eval_transform: List[Callable] = None,
    ):
        """Initialization, taking care of HDF5 dataset preparation if needed, and indexation of its content.

        Args:
            las_files_by_split (Optional[LAS_FILES_BY_SPLIT_TYPE]): should look like
                las_files_by_split = {'train': ['las1.las','las2.las'], 'val': [...], , 'test': [...]}
            hdf5_file_path (str): path to HDF5 dataset
            points_pre_transform (Callable): Function to turn pdal points into a pyg Data object.
            tile_width (Number, optional): width of a LAS tile. Defaults to 1000.
            subtile_width (Number, optional): effective width of a subtile (i.e. receptive field). Defaults to 50.
            subtile_shape (SHAPE_TYPE, optional): Shape of subtile could be either "square" or "disk". Defaults to "square".
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
        self.subtile_shape = subtile_shape

        self.hdf5_file_path = hdf5_file_path

        # Instantiates these to null;
        # They are loaded within __getitem__ to support multi-processing training.
        self.dataset = None
        self.samples_hdf5_paths = None

        if las_files_by_split is None:
            log.warning(
                "las_files_by_split is None and pre-computed HDF5 dataset is therefore used."
            )
            return

        # Add data for all LAS Files into a single hdf5 file.
        os.makedirs(osp.dirname(hdf5_file_path), exist_ok=True)
        for split, las_paths in las_files_by_split.items():
            with h5py.File(hdf5_file_path, "a") as f:
                if split not in f:
                    f.create_group(split)
            for las_path in tqdm(las_paths, desc=f"Preparing {split} set..."):
                self._add_to_dataset(split, las_path)

        self._index_all_samples_as_needed()

    def _add_to_dataset(self, split: str, las_path: str):
        """Add samples from LAS into HDF5 dataset file.

        Args:
            split (str): specifies either "train", "val", or "test" split.
            las_path (str): path to point cloud.

        """
        b = osp.basename(las_path)

        # Delete dataset for incomplete LAS entry, to start from scratch.
        # Useful in case data preparation was interrupted.
        with h5py.File(self.hdf5_file_path, "a") as f:
            if b in f[split] and "is_complete" not in f[split][b].attrs:
                del f[b]
                # Parse and add subtiles to split group.
        with h5py.File(self.hdf5_file_path, "a") as f:
            if b not in f[split]:
                sample_number = 0
                for sample_idx, sample_points in split_cloud_into_samples(
                    las_path,
                    self.tile_width,
                    self.subtile_width,
                    self.subtile_shape,
                    # No overlap at eval time.
                    subtile_overlap=self.subtile_overlap_train
                    if split == "train"
                    else 0,
                ):
                    data = self.points_pre_transform(sample_points)
                    if self.pre_filter is not None and self.pre_filter(data):
                        # e.g. pre_filter spots situations where num_nodes is too small.
                        continue
                    sample_id = str(sample_number).zfill(5)
                    hdf5_path = osp.join(split, b, sample_id)
                    hd5f_path_x = osp.join(hdf5_path, "x")
                    f.create_dataset(
                        hd5f_path_x,
                        data.x.shape,
                        dtype="f",
                        data=data.x,
                    )
                    f[hd5f_path_x].attrs["x_features_names"] = copy.deepcopy(
                        data.x_features_names
                    )
                    f.create_dataset(
                        osp.join(hdf5_path, "pos"),
                        data.pos.shape,
                        dtype="f",
                        data=data.pos,
                    )
                    f.create_dataset(
                        osp.join(hdf5_path, "y"),
                        data.y.shape,
                        dtype="i",
                        data=data.y,
                    )
                    f.create_dataset(
                        osp.join(hdf5_path, "idx_in_original_cloud"),
                        sample_idx.shape,
                        dtype="i",
                        data=sample_idx,
                    )
                    sample_number += 1

                # A termination flag to report that all samples for this point cloud were included in the df5 file.
                f[split][b].attrs["is_complete"] = True

    def _index_all_samples_as_needed(self):
        """Index all samples in the dataset, if not already done before."""
        # Use existing if already indexed. Need to decode b-string
        if self.samples_hdf5_paths is not None:
            return

        # Use stored if already indexed. Need to decode b-string
        with h5py.File(self.hdf5_file_path, "r") as f:
            if "samples_hdf5_paths" in f:
                self.samples_hdf5_paths = [
                    sample_path.decode("utf-8")
                    for sample_path in f["samples_hdf5_paths"]
                ]
                return

        # Otherwise, index samples, and add the index as an attribute to the HDF5 file.
        self.samples_hdf5_paths = []
        with h5py.File(self.hdf5_file_path, "r") as f:
            for s in [s for s in f.keys() if s in ["train", "val", "test"]]:
                basenames = f[s].keys()
                for bn in basenames:
                    self.samples_hdf5_paths += [
                        osp.join(s, bn, sample_number)
                        for sample_number in f[s][bn].keys()
                    ]
        with h5py.File(self.hdf5_file_path, "a") as f:
            variable_lenght_str_datatype = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "samples_hdf5_paths",
                (len(self.samples_hdf5_paths),),
                dtype=variable_lenght_str_datatype,
                data=self.samples_hdf5_paths,
            )

    def __len__(self):
        return len(self.samples_hdf5_paths)

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
            # num_nodes=grp["pos"][...].shape[0],  # Not needed - performed under the hood.
            x_features_names=grp["x"].attrs["x_features_names"].tolist(),
        )

    def __getitem__(self, idx: int) -> Optional[Data]:
        self._index_all_samples_as_needed()
        sample_hdf5_path = self.samples_hdf5_paths[idx]
        data = self._get_data(sample_hdf5_path)

        # filter if empty
        if self.pre_filter is not None and self.pre_filter(data):
            return None

        # Transforms, including sampling and some augmentations.
        transform = self.train_transform
        if sample_hdf5_path.startswith("val") or sample_hdf5_path.startswith(
            "test"
        ):
            transform = self.eval_transform
        if transform is not None:
            data = transform(data)

        # filter if empty
        if data is None or (
            self.pre_filter is not None and self.pre_filter(data)
        ):
            return None

        return data

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
        self._index_all_samples_as_needed()
        indices = [
            idx
            for idx, p in enumerate(self.samples_hdf5_paths)
            if p.startswith(split)
        ]
        return torch.utils.data.Subset(self, indices)
