import os
import os.path as osp
from abc import abstractmethod
from numbers import Number
from typing import List

import numpy as np
import pdal
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from .utils import (
    find_file_in_dir,
    get_mosaic_of_centers,
    get_random_center_in_tile,
    make_circle_wkt,
)


class COPCDataset(Dataset):
    """Dataset for data augmentation of large LAS tiles, for deep learning training/inference, using COPC format.
    See https://lidarmag.com/2021/12/27/cloud-native-geospatial-lidar-with-the-cloud-optimized-point-cloud/ for more
    details.

    Nota: the related DataModule is not implemented at the moment.
    There is a need to validate speed/performance first. Right now, it is not fast enough to support
    large batch loading for deep learning applications. LAZ decompression occuring in COPC might be a bottleneck.
    """

    def __init__(
        self,
        tiles_basenames: List[str],
        copc_dir,
        data_dir=None,
        add_original_index: bool = True,
    ):
        if len(tiles_basenames) == 0:
            raise KeyError("Given list of files is empty")

        processed_basenames = [b.replace(".las", ".copc.laz") for b in tiles_basenames]
        self.copc_paths = [osp.join(copc_dir, b) for b in processed_basenames]

        if data_dir:
            # CONVERSION TO COPC IF NEEDED
            raw_paths = [find_file_in_dir(data_dir, b) for b in tiles_basenames]
            try:
                # IndexError if no file is found in dir.
                [find_file_in_dir(copc_dir, b) for b in processed_basenames]
            except IndexError:
                # some processed file are not created yet in processed_dir
                os.makedirs(copc_dir, exist_ok=True)
                for las_path, copc_laz_path in tqdm(
                    zip(raw_paths, self.copc_paths),
                    desc="Conversion to COPC.LAZ format.",
                ):
                    write_las_to_copc_laz(
                        las_path,
                        copc_laz_path,
                        add_original_index=add_original_index,
                    )

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def load_points(idx) -> np.ndarray:
        raise NotImplementedError()

    def __getitem__(self, idx):
        points = self.load_points(idx)

        # filter if empty
        if len(points) == 0:
            return None

        # Turn into a pytorch_geometric Data object.
        data: Data = self.points_pre_transform(points)
        for attr in ["x", "pos", "y"]:
            data[attr] = torch.from_numpy(data[attr])

        # filter if empty
        if self.pre_filter is not None and self.pre_filter(data):
            return None

        # Transforms, including sampling and some augmentations.
        if self.transform is not None:
            data = self.transform(data)

        # filter if empty
        if data is None or (self.pre_filter is not None and self.pre_filter(data)):
            return None

        return data

    def visualize_sample(self, idx):
        print(self[idx])


class COPCRandomDataset(COPCDataset):
    """Dataset for random selection of subtile in large LAS tiles, for deep learning training."""

    def __init__(
        self,
        tiles_basenames: List[str],
        copc_dir,  # like /path/to/root/val/
        datadir=None,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        points_pre_transform=None,
        transform=None,
        pre_filter=None,
        subtile_by_tile_at_each_epoch: Number = 1,
        resolution: float = 0.0,
    ):
        super().__init__(
            tiles_basenames,
            copc_dir,
            data_dir=datadir,
            add_original_index=False,
        )

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.resolution = resolution

        self.points_pre_transform = points_pre_transform
        self.transform = transform
        self.pre_filter = pre_filter

        if subtile_by_tile_at_each_epoch > 1:
            # Load more than one subtile for each tile.
            # Useful when dealing with n files with n<batch_size.
            self.copc_paths = subtile_by_tile_at_each_epoch * self.copc_paths

    def __len__(self):
        # One epoch = extract one subtile from each large tile.
        return len(self.copc_paths)

    def load_points(self, idx) -> np.ndarray:
        copc_path = self.copc_paths[idx]
        center = get_random_center_in_tile(self.tile_width, self.subtile_width)
        wkt = make_circle_wkt(center, self.subtile_width)
        points = load_from_copc(copc_path, polygon=wkt, resolution=self.resolution)
        return points


class COPCInferenceDataset(COPCDataset):
    """Dataset for inference."""

    def __init__(
        self,
        tiles_basenames: List[str],
        copc_dir,  # like /path/to/root/val/
        data_dir="",
        transform=None,
        points_pre_transform=None,
        pre_filter=None,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        add_original_index: bool = True,
        resolution: float = 0.0,
    ):
        super().__init__(
            tiles_basenames,
            copc_dir,
            data_dir=data_dir,
            add_original_index=add_original_index,
        )

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.resolution = resolution

        self.points_pre_transform = points_pre_transform
        self.transform = transform
        self.pre_filter = pre_filter

        # samples is a list of path-center pairs
        xy_centers = get_mosaic_of_centers(
            self.tile_width,
            self.subtile_width,
            subtile_overlap=subtile_overlap,
        )
        self.samples = []
        for path in self.copc_paths:
            for xy_center in xy_centers:
                self.samples += [(path, xy_center)]

    def __len__(self):
        # One epoch = all samples from all files
        return len(self.samples)

    def load_points(self, idx) -> np.ndarray:
        copc_path, center = self.samples[idx]
        wkt = make_circle_wkt(center, self.subtile_width)
        points = load_from_copc(copc_path, polygon=wkt)
        return points


class COPCEvalDataset(COPCInferenceDataset):
    """Dataset for evaluation.

    Extract a mosaic of subtiles that cover the entire input tiles.
    Similar to COPCInferenceDataset except that there subtile overlap is set to 0
    and no extra index dimension is created.

    """

    def __init__(
        self,
        tiles_basenames: List[str],
        copc_dir,  # like /path/to/root/val/
        data_dir="",
        transform=None,
        points_pre_transform=None,
        pre_filter=None,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        resolution: float = 0.0,
    ):
        super().__init__(
            tiles_basenames,
            copc_dir,
            data_dir=data_dir,
            transform=transform,
            points_pre_transform=points_pre_transform,
            pre_filter=pre_filter,
            tile_width=tile_width,
            subtile_width=subtile_width,
            subtile_overlap=0,
            add_original_index=False,
            resolution=resolution,
        )


def write_las_to_copc_laz(las_path: str, copc_laz_path: str, add_original_index: bool = False):
    """Convert from LAS to COPC, for optimized later loading.

    Resulting data starts at 0 on x and y.

    Args:
        las_path (str): _description_
        copc_laz_path (str): _description_
        min_normalize (bool): wether to offset x and y dims by their minimal value.

    Returns:
        _type_: _description_
    """
    reader = pdal.Pipeline() | pdal.Reader.las(
        filename=las_path, nosrs=True, override_srs="EPSG:2154"
    )
    if add_original_index:
        reader |= pdal.Filter.ferry("=>OriginalIndex")
    reader.execute()
    points = reader.arrays[0]
    if add_original_index:
        points["OriginalIndex"] = np.arange(len(points))
    points["X"] = points["X"] - points["X"].min()
    points["Y"] = points["Y"] - points["Y"].min()
    writer = pdal.Writer.copc(copc_laz_path, forward="all").pipeline(points)
    writer.execute()


def load_from_copc(copc_laz_path: str, **kwargs) -> np.ndarray:
    """Load from copc.laz file, specifying area via kwargs."""
    pipeline = pdal.Pipeline() | pdal.Reader.copc(
        copc_laz_path,
        **kwargs,
    )
    pipeline.execute()
    return pipeline.arrays[0]
