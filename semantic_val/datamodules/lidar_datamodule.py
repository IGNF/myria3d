import glob
import os.path as osp
from typing import Optional, Union

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, NormalizeScale, RandomFlip
from torch_geometric.transforms.compose import Compose

from semantic_val.datamodules.datasets.lidar_dataset import (
    LidarToyTestDataset,
    LidarTrainDataset,
    LidarValDataset,
)
from semantic_val.datamodules.datasets.lidar_utils import (
    collate_fn,
    transform_labels_for_building_segmentation,
)


class KeepOriginalPos(BaseTransform):
    r"""Make a copy of unormalized positions."""

    def __call__(self, data: Data):
        data["origin_pos"] = data["pos"].copy()
        return data


class ToTensor(BaseTransform):
    r"""Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys=["x", "origin_pos", "pos", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


class LidarDataModule(LightningDataModule):
    """
    Nota: we do not collate cloud in order to feed full cloud of various size to models directly,
    so they can give full outputs for evaluation and inference.
    """

    def __init__(
        self,
        data_dir: str = "./data/lidar_toy/",
        batch_size: int = 8,
        num_workers: int = 0,
        subtile_width_meters: float = 100.0,
        subtile_overlap: float = 0.0,
        train_subtiles_by_tile: int = 4,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subtile_width_meters = subtile_width_meters
        self.train_subtiles_by_tile = train_subtiles_by_tile

        self.subtile_overlap = subtile_overlap
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        # TODO: implement train-val-test split that add a split column used as reference for later datasets
        pass

    def get_train_transforms(self) -> Compose:
        """Create a transform composition for train phase."""
        return Compose(
            [
                KeepOriginalPos(),
                ToTensor(),
                NormalizeScale(),
                transform_labels_for_building_segmentation,
                # TODO: set data augmentation back when overfitting is possible.
                # RandomFlip(0, p=0.5),
                # RandomFlip(1, p=0.5),
            ]
        )

    def get_val_transforms(self) -> Compose:
        """Create a transform composition for val phase."""
        return Compose(
            [
                KeepOriginalPos(),
                ToTensor(),
                NormalizeScale(),
                transform_labels_for_building_segmentation,
            ]
        )

    def get_test_transforms(self) -> Compose:
        return self.get_val_transforms()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        train_files = glob.glob(osp.join(self.data_dir, "train/*.las"))
        train_files = sorted(train_files * self.train_subtiles_by_tile)
        val_files = glob.glob(osp.join(self.data_dir, "val/*.las"))
        test_files = glob.glob(osp.join(self.data_dir, "test/*.las"))

        self.data_train = LidarTrainDataset(
            train_files,
            transform=self.get_train_transforms(),
            target_transform=transform_labels_for_building_segmentation,
            subtile_width_meters=self.subtile_width_meters,
        )
        # self.dims = tuple(self.data_train[0].x.shape)
        self.data_val = LidarValDataset(
            val_files,
            transform=self.get_val_transforms(),
            target_transform=transform_labels_for_building_segmentation,
            subtile_width_meters=self.subtile_width_meters,
            subtile_overlap=self.subtile_overlap,
        )
        self.data_test = LidarToyTestDataset(
            test_files,
            transform=self.get_test_transforms(),
            target_transform=transform_labels_for_building_segmentation,
            subtile_width_meters=self.subtile_width_meters,
            subtile_overlap=self.subtile_overlap,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
