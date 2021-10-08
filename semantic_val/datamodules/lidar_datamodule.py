import glob
import os.path as osp
from typing import Optional, Union
import pandas as pd

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, NormalizeScale, RandomFlip
from torch_geometric.transforms.compose import Compose


from semantic_val.datamodules.datasets.lidar_dataset import (
    LidarTestDataset,
    LidarTrainDataset,
    LidarValDataset,
)
from semantic_val.datamodules.datasets.lidar_utils import (
    collate_fn,
    create_full_filepath_column,
    get_metadata_df_from_shapefile,
    get_random_subtile_center,
    get_split_df_of_202110_building_val,
    get_subtile_data,
    get_tile_center,
)
from semantic_val.utils import utils

log = utils.get_logger(__name__)


class SelectSubTile(BaseTransform):
    r"""Select a square subtile from original tile"""

    def __init__(
        self,
        subtile_width_meters: float = 100.0,
        method=["deterministic", "predefined", "random"],
    ):
        self.subtile_width_meters = subtile_width_meters
        self.method = method

    def __call__(self, data: Data):

        for try_i in range(25):
            if self.method == "deterministic":
                center = get_tile_center(data, self.subtile_width_meters)
            elif self.method == "random":
                center = get_random_subtile_center(
                    data, subtile_width_meters=self.subtile_width_meters
                )
            elif self.method == "predefined":
                center = data.current_subtile_center
            else:
                raise f"Undefined method argument: {self.method}"
            subtile_data = get_subtile_data(
                data,
                center,
                subtile_width_meters=self.subtile_width_meters,
            )
            if subtile_data.pos.shape[0] == 0:
                log.info(
                    f"Error - no points in subtile extracted from {data.filepath} at position {str(center)}"
                )
                log.info(f"New try of a random extract (i={try_i+1}/10)")
            else:
                break
        return subtile_data


class ToTensor(BaseTransform):
    r"""Turn np.arrays specified by their keys into Tensor."""

    def __init__(self, keys=["pos", "x", "y"]):
        self.keys = keys

    def __call__(self, data: Data):
        for key in data.keys:
            if key in self.keys:
                data[key] = torch.from_numpy(data[key])
        return data


class KeepOriginalPos(BaseTransform):
    r"""Make a copy of unormalized positions."""

    def __call__(self, data: Data):
        data["origin_pos"] = data["pos"].clone()
        return data


class NormalizeFeatures(BaseTransform):
    r"""Scale features in 0-1 range."""

    def __call__(self, data: Data):
        INTENSITY_IDX = 0
        RETURN_NUM_IDX = 1
        NUM_RETURN_IDX = 2

        INTENSITY_MAX = 32768.0
        RETURN_NUM_MAX = 7

        data["x"][:, INTENSITY_IDX] = data["x"][:, INTENSITY_IDX] / INTENSITY_MAX
        data["x"][:, RETURN_NUM_IDX] = (data["x"][:, RETURN_NUM_IDX] - 1) / (
            RETURN_NUM_MAX - 1
        )
        data["x"][:, NUM_RETURN_IDX] = (data["x"][:, NUM_RETURN_IDX] - 1) / (
            RETURN_NUM_MAX - 1
        )
        return data


class MakeBuildingTargets(BaseTransform):
    """
    Pass from multiple classes to simpler Building/Non-Building labels.
    Initial classes: [  1,   2,   6 (detected building, no validation),  19 (valid building),  20 (surdetection, unspecified),
    21 (building, forgotten), 104, 110 (surdetection, others), 112 (surdetection, vehicule), 114 (surdetection, others), 115 (surdetection, bridges)]
    Final classes: 0 (non-building), 1 (building)
    """

    def __call__(self, data: Data):
        buildings_idx = (data.y == 19) | (data.y == 21) | (data.y == 6)
        data.y[buildings_idx] = 1
        data.y[~buildings_idx] = 0
        return data


class LidarDataModule(LightningDataModule):
    """
    Nota: we do not collate cloud in order to feed full cloud of various size to models directly,
    so they can give full outputs for evaluation and inference.
    """

    def __init__(
        self,
        lasfiles_dir: str = "./path/to/dataset/trainvaltest/",
        metadata_shapefile: str = "./path/to/dataset/my_shapefile.shp",
        train_frac: float = 0.8,
        batch_size: int = 8,
        num_workers: int = 0,
        subtile_width_meters: float = 100.0,
        subtile_overlap: float = 0.0,
        train_subtiles_by_tile: int = 4,
        overfit: bool = False,
    ):
        super().__init__()

        self.num_workers = num_workers

        self.lasfiles_dir = lasfiles_dir
        self.metadata_shapefile = metadata_shapefile
        self.datasplit_csv_filepath = metadata_shapefile.replace(".shp", ".csv")

        self.overfit = overfit
        self.train_frac = train_frac
        self.subtile_width_meters = subtile_width_meters
        self.train_subtiles_by_tile = train_subtiles_by_tile
        self.subtile_overlap = subtile_overlap
        self.batch_size = batch_size

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def make_datasplit_csv(
        self, shapefile_filepath, datasplit_csv_filepath, train_frac=0.8
    ):
        """Turn the shapefile of tiles metadata into a csv with stratified train-val-test split."""
        df = get_metadata_df_from_shapefile(shapefile_filepath)
        df_split = get_split_df_of_202110_building_val(df, train_frac=train_frac)
        df_split = create_full_filepath_column(df_split, self.lasfiles_dir)
        assert all(osp.exists(filepath) for filepath in df_split.file_path)
        df_split.to_csv(datasplit_csv_filepath, index=False)

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""

        las_filepaths = glob.glob(osp.join(self.lasfiles_dir, "*.las"))
        assert len(las_filepaths) == 150

        if not osp.exists(self.datasplit_csv_filepath):
            if not osp.exists(self.metadata_shapefile):
                raise FileNotFoundError(
                    f"Data-descriptive shapefile not found at {self.metadata_shapefile}"
                )
            self.make_datasplit_csv(
                self.metadata_shapefile,
                self.datasplit_csv_filepath,
                train_frac=self.train_frac,
            )
            log.info(
                f"Stratified split of dataset saved to {self.datasplit_csv_filepath}"
            )

    def get_train_transforms(self) -> Compose:
        """Create a transform composition for train phase."""
        return Compose(
            [
                SelectSubTile(
                    subtile_width_meters=self.subtile_width_meters,
                    method="deterministic" if self.overfit else "random",
                ),
                ToTensor(),
                KeepOriginalPos(),
                NormalizeFeatures(),
                NormalizeScale(),
                # TODO: set data augmentation back when regularization is needed.
                # RandomFlip(0, p=0.5),
                # RandomFlip(1, p=0.5),
            ]
        )

    def get_val_transforms(self) -> Compose:
        """Create a transform composition for val phase."""
        return Compose(
            [
                SelectSubTile(
                    subtile_width_meters=self.subtile_width_meters, method="predefined"
                ),
                ToTensor(),
                KeepOriginalPos(),
                NormalizeFeatures(),
                NormalizeScale(),
            ]
        )

    def get_test_transforms(self) -> Compose:
        return self.get_val_transforms()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        df_split = pd.read_csv(self.datasplit_csv_filepath)

        train_files = df_split[df_split.split == "train"].file_path.values.tolist()
        if self.overfit:
            train_files = [
                filepath
                for filepath in train_files
                if filepath.endswith("845000_6610000.las")
            ]
        train_files = sorted(train_files * self.train_subtiles_by_tile)
        val_files = df_split[df_split.split == "val"].file_path.values.tolist()
        test_files = df_split[df_split.split == "test"].file_path.values.tolist()

        self.data_train = LidarTrainDataset(
            train_files,
            transform=self.get_train_transforms(),
            target_transform=MakeBuildingTargets(),
            subtile_width_meters=self.subtile_width_meters,
        )
        # self.dims = tuple(self.data_train[0].x.shape)
        self.data_val = LidarValDataset(
            val_files,
            transform=self.get_val_transforms(),
            target_transform=MakeBuildingTargets(),
            subtile_width_meters=self.subtile_width_meters,
            subtile_overlap=self.subtile_overlap,
        )
        self.data_test = LidarTestDataset(
            test_files,
            transform=self.get_test_transforms(),
            target_transform=MakeBuildingTargets(),
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
