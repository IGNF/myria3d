import glob
import os
import os.path as osp
import random
from typing import Optional, Iterator, Optional
import itertools

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch_geometric.transforms import (
    RandomRotate,
    RandomTranslate,
    RandomScale,
    RandomFlip,
)
from torch_geometric.transforms.center import Center

from semantic_val.datamodules.datasets.SemValBuildings202110 import (
    LidarTrainDataset,
    LidarValDataset,
    make_datasplit_csv,
)
from semantic_val.datamodules.processing import *
from semantic_val.utils import utils

log = utils.get_logger(__name__)


class DataModule(LightningDataModule):
    """
    Datamdule to feed train and validation data to the model.
    We use a custome sampler during training to consecutively load several
    subtiles from a single tile, to reduce the I/O footprint.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.num_workers = kwargs.get("num_workers", 0)

        self.lasfiles_dir = kwargs.get("lasfiles_dir")
        self.metadata_shapefile_filepath = kwargs.get("metadata_shapefile")
        self.datasplit_csv_filepath = kwargs.get("metadata_shapefile").replace(
            ".shp", ".csv"
        )

        self.train_frac = kwargs.get("train_frac", 0.8)
        self.shuffle_train = kwargs.get("shuffle_train", "true")
        self.limit_top_k_tiles_train = kwargs.get("limit_top_k_tiles_train", 1000000)
        self.limit_top_k_tiles_val = kwargs.get("limit_top_k_tiles_val", 1000000)
        self.train_subtiles_by_tile = kwargs.get("train_subtiles_by_tile", 12)
        self.batch_size = kwargs.get("batch_size", 32)
        self.augment = kwargs.get("augment", True)

        self.subtile_width_meters = kwargs.get("subtile_width_meters", 50)
        self.subtile_overlap = kwargs.get("subtile_overlap", 0)
        self.subsample_size = kwargs.get("subsample_size", 12500)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def prepare_data(self):
        """
        Stratify train/val/test data if needed.
        Nota: Do not use it to assign state (self.x = y). This method is called only from a single GPU.
        """
        if osp.exists(self.datasplit_csv_filepath):
            os.remove(self.datasplit_csv_filepath)

<<<<<<< ours
        # las_filepaths = glob.glob(osp.join(self.lasfiles_dir, "*.las"))
        # assert len(las_filepaths) == 150

        if not osp.exists(self.datasplit_csv_filepath):
            if not osp.exists(self.metadata_shapefile_filepath):
                raise FileNotFoundError(
                    f"Metadata shapefile not found at {self.metadata_shapefile_filepath}"
                )
            make_datasplit_csv(
                self.lasfiles_dir,
                self.metadata_shapefile_filepath,
                self.datasplit_csv_filepath,
                train_frac=self.train_frac,
            )
            log.info(
                f"Stratified split of dataset saved to {self.datasplit_csv_filepath}"
=======
        if not osp.exists(self.metadata_shapefile_filepath):
            raise FileNotFoundError(
                f"Metadata shapefile not found at {self.metadata_shapefile_filepath}"
>>>>>>> theirs
            )
        make_datasplit_csv(
            self.lasfiles_dir,
            self.metadata_shapefile_filepath,
            self.datasplit_csv_filepath,
            train_frac=self.train_frac,
        )
        log.info(
            f"Stratified split of dataset saved to {self.datasplit_csv_filepath}"
        )

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.data_train, self.data_val, self.data_test.
        test_data = val data, because we only use all validation data after training.
        Test data can be used but only after final model is chosen.
        """

        self._set_all_transforms()

        df_split = pd.read_csv(self.datasplit_csv_filepath)

        self._set_train_data(df_split)
        self._set_val_data(df_split)
        self._set_test_data(df_split)

    def _set_train_data(self, df_split):
        """Get the train dataset"""
        df_split_train = df_split[df_split.split == "train"]
        df_split_train = df_split_train.sort_values("nb_bati", ascending=False)
        train_files = df_split_train.file_path.values.tolist()
        if self.limit_top_k_tiles_train:
            train_files = train_files[: self.limit_top_k_tiles_train]
            log.info(
                "\n Training on: \n " + str([osp.basename(f) for f in train_files])
            )
        self.train_data = LidarTrainDataset(
            train_files,
            loading_function=load_las_data,
            transform=self._get_train_transforms(),
            target_transform=MakeBuildingTargets(),
            subtile_width_meters=self.subtile_width_meters,
            train_subtiles_by_tile=self.train_subtiles_by_tile,
        )

    def _set_val_data(self, df_split):
        """Get the validation dataset"""
        df_split_val = df_split[df_split.split == "val"]
        df_split_val = df_split_val.sort_values("nb_bati", ascending=False)
        val_files = df_split_val.file_path.values.tolist()
        if self.limit_top_k_tiles_val:
            val_files = val_files[: self.limit_top_k_tiles_val]
            log.info(
                "\n Validating on: \n " + str([osp.basename(f) for f in val_files])
            )
        self.val_data = LidarValDataset(
            val_files,
            loading_function=load_las_data,
            transform=self._get_val_transforms(),
            target_transform=MakeBuildingTargets(),
            subtile_width_meters=self.subtile_width_meters,
            subtile_overlap=self.subtile_overlap,
        )

    def _set_test_data(self, df_split):
        """Get the test dataset - for now this is the validation dataset"""
        self.test_data = self.val_data

    def train_dataloader(self):
        """Get train dataloader."""
        sampler = TrainSampler(
            self.train_data.nb_files,
            self.train_subtiles_by_tile,
            shuffle_train=self.shuffle_train,
        )
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Get val dataloader. num_workers is only one because load must be split for IterableDataset.
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )

    def _set_all_transforms(self):
        """
        Set transforms that are shared between train/val-test.
        Called at initialization.
        """

        MAX_ROTATION_DEGREES = 5
        MIN_RANDOM_SCALE = 0.9
        MAX_RANDOM_SCALE = 1.1
        POS_TRANSLATIONS_METERS = (0.25, 0.25, 0.25)

        self.preparation = [
            ToTensor(),
            MakeCopyOfPosAndY(),
            FixedPointsPosXY(self.subsample_size, replace=False, allow_duplicates=True),
            MakeCopyOfSampledPos(),
            Center(),
        ]
        # TODO: add a 90° rotation using LinearTransformation nested in a custom transforms
        self.augmentation = []
        if self.augment:
            self.augmentation = [
                RandomFlip(0, p=0.5),
                RandomFlip(1, p=0.5),
                RandomRotate(MAX_ROTATION_DEGREES, axis=2),
                RandomTranslate(POS_TRANSLATIONS_METERS),
                RandomScale((MIN_RANDOM_SCALE, MAX_RANDOM_SCALE)),
                RandomTranslateFeatures(),
            ]
        self.normalization = [
            CustomNormalizeScale(),
            CustomNormalizeFeatures(),
        ]

    def _get_train_transforms(self) -> CustomCompose:
        """Create a transform composition for train phase."""
        selection = SelectSubTile(
            subtile_width_meters=self.subtile_width_meters,
            method="random",
        )

        return CustomCompose(
            [selection] + self.preparation + self.augmentation + self.normalization
        )

    def _get_val_transforms(self) -> CustomCompose:
        """Create a transform composition for val phase."""
        selection = SelectSubTile(
            subtile_width_meters=self.subtile_width_meters, method="predefined"
        )

        return CustomCompose([selection] + self.preparation + self.normalization)

    def _get_test_transforms(self) -> CustomCompose:
        return self._get_val_transforms()


class TrainSampler(Sampler[int]):
    """Custom sampler to draw multiple subtiles from a file in the same batch."""

    def __init__(
        self, nb_files: int, train_subtiles_by_tile: int, shuffle_train: bool = False
    ) -> None:
        """:param data_source_size: Number of training LAS files."""
        self.nb_files = nb_files
        self.data_source_range = list(range(self.nb_files))
        self.train_subtiles_by_tile = train_subtiles_by_tile
        self.shuffle_train = shuffle_train

    def __iter__(self) -> Iterator[int]:
        """Shuffle the files query indexes, and n-plicate them while keeping their new order."""
        if self.shuffle_train:
            random.shuffle(self.data_source_range)
        extended_range = [
            ([file_idx] * self.train_subtiles_by_tile)
            for file_idx in self.data_source_range
        ]
        return itertools.chain.from_iterable(extended_range)

    def __len__(self) -> int:
        return self.nb_files * self.train_subtiles_by_tile
