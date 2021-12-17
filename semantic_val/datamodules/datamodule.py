import ast
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
    SPLIT_LAS_DIR_COLN,
    LidarMapDataset,
    LidarIterableDataset,
    make_datasplit_csv,
)
from semantic_val.datamodules.processing import *
from semantic_val.decision.codes import MTS_AUTO_DETECTED_CODE
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
        self.limit_top_k_tiles_train = kwargs.get("limit_top_k_tiles_train", 1000000)
        self.limit_top_k_tiles_val = kwargs.get("limit_top_k_tiles_val", 1000000)
        self.train_subtiles_by_tile = kwargs.get("train_subtiles_by_tile", 12)
        self.batch_size = kwargs.get("batch_size", 32)
        self.augment = kwargs.get("augment", True)

        # TODO: remove if we also preprocess in predict mode.
        self.subtile_width_meters = kwargs.get("subtile_width_meters", 50)
        self.subsample_size = kwargs.get("subsample_size", 12500)

        self.src_las = kwargs.get("src_las", None)  # predict#
        # By default, do not use the test set unless explicitely required by user.
        self.use_val_data_at_test_time = kwargs.get("use_val_data_at_test_time", True)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.predict_data: Optional[Dataset] = None  # predict#

    def prepare_data(self):
        """
        Stratify train/val/test data if needed.
        Nota: Do not use it to assign state (self.x = y). This method is called only from a single GPU.
        """
        if (
            self.trainer.state.stage == "predict"
        ):  # no creation of CSV file if it's an inference
            return

        if not osp.exists(self.metadata_shapefile_filepath):
            raise FileNotFoundError(
                f"Metadata shapefile not found at {self.metadata_shapefile_filepath}"
            )
        if osp.exists(self.datasplit_csv_filepath):
            os.remove(self.datasplit_csv_filepath)
        make_datasplit_csv(
            self.lasfiles_dir,
            self.metadata_shapefile_filepath,
            self.datasplit_csv_filepath,
            train_frac=self.train_frac,
        )
        log.info(f"Stratified split of dataset saved to {self.datasplit_csv_filepath}")

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.data_train, self.data_val, self.data_test.
        test_data = val data, because we only use all validation data after training.
        Test data can be used but only after final model is chosen.
        """
        self._set_all_transforms()

        if stage != "predict":
            df_split = pd.read_csv(
                self.datasplit_csv_filepath,
                converters={SPLIT_LAS_DIR_COLN: ast.literal_eval},
            )
            self._set_train_data(df_split)
            self._set_val_data(df_split)
            self._set_test_data(df_split)

    def _set_train_data(self, df_split):
        """Get the train dataset"""
        df = df_split[df_split.split == "train"]
        df = df.sort_values("nb_bati", ascending=False)
        f_lists = df[SPLIT_LAS_DIR_COLN].values.tolist()
        if self.limit_top_k_tiles_train < len(f_lists):
            f_lists = f_lists[: self.limit_top_k_tiles_train]
            log.info(f"Training on {self.limit_top_k_tiles_train}) tiles.")
        files = [f for l in f_lists for f in l]
        self.train_data = LidarMapDataset(
            files,
            loading_function=load_las_data,
            transform=self._get_train_transforms(),
            target_transform=MakeBuildingTargets(),
        )

    def _set_val_data(self, df_split):
        """Get the validation dataset"""
        df = df_split[df_split.split == "val"]
        df = df.sort_values("nb_bati", ascending=False)
        files_lists = df[SPLIT_LAS_DIR_COLN].values.tolist()
        if self.limit_top_k_tiles_val:
            files_lists = files_lists[: self.limit_top_k_tiles_train]
        log.info(f"Validation on {len(files_lists)}) tiles.")
        self.val_data = [
            LidarMapDataset(
                files,
                loading_function=load_las_data,
                transform=self._get_val_transforms(),
                target_transform=MakeBuildingTargets(),
            )
            for files in files_lists
        ]

    def _set_test_data(self, df_split):
        """Get the test dataset. User need to explicitely require the use of test set, which is kept out of experiment until the end."""
        if self.use_val_data_at_test_time:
            self.test_data = self.val_data
            log.info(
                "Using validation data as test data. Use real test data with use_val_data_at_test_time=False at run time."
            )
            return

        df = df_split[df_split.split == "test"]
        df = df.sort_values("nb_bati", ascending=False)
        files_lists = df[SPLIT_LAS_DIR_COLN].values.tolist()
        # One dataset per cloud
        self.test_data = [
            LidarMapDataset(
                files,
                loading_function=load_las_data,
                transform=self._get_test_transforms(),
                target_transform=MakeBuildingTargets(),
            )
            for files in files_lists
        ]

    def _set_predict_data(
        self, files_to_infer_on, mts_auto_detected_code: int = MTS_AUTO_DETECTED_CODE
    ):
        self.predict_data = LidarIterableDataset(
            files_to_infer_on,
            loading_function=load_las_data,
            transform=self._get_predict_transforms(mts_auto_detected_code),
            target_transform=None,
            subtile_width_meters=self.subtile_width_meters,
        )

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        """
        Get val dataloader. num_workers is only one because load must be split for IterableDataset.
        """
        return [
            DataLoader(
                dataset=tile_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            )
            for tile_dataset in self.val_data
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                dataset=tile_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
            )
            for tile_dataset in self.test_data
        ]

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_data,
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
            EmptySubtileFilter(),
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
        return CustomCompose(self.preparation + self.augmentation + self.normalization)

    def _get_val_transforms(self) -> CustomCompose:
        """Create a transform composition for val phase."""
        return CustomCompose(self.preparation + self.normalization)

    def _get_test_transforms(self) -> CustomCompose:
        return self._get_val_transforms()

    def _get_predict_transforms(
        self, mts_auto_detected_code: int = MTS_AUTO_DETECTED_CODE
    ) -> CustomCompose:
        """Create a transform composition for predict phase."""
        selection = SelectPredictSubTile(
            subtile_width_meters=self.subtile_width_meters,
            mts_auto_detected_code=mts_auto_detected_code,
        )
        return CustomCompose([selection] + self.preparation + self.normalization)
