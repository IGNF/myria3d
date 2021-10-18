import glob
import os.path as osp
import random
from typing import Optional, Union, Iterator, Optional, List, Sized
import itertools

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch_geometric.transforms.compose import Compose
from torch_geometric.transforms import (
    NormalizeScale,
    RandomFlip,
)

from semantic_val.datamodules.datasets.lidar_dataset import (
    LidarTestDataset,
    LidarTrainDataset,
    LidarValDataset,
)
from semantic_val.datamodules.datasets.lidar_transforms import (
    FixedPointsPosXY,
    MakeBuildingTargets,
    MakeCopyOfPosAndY,
    MakeCopyOfSampledPos,
    NormalizeFeatures,
    CustomNormalizeScale,
    SelectSubTile,
    ToTensor,
    collate_fn,
    create_full_filepath_column,
    get_metadata_df_from_shapefile,
    get_split_df_of_202110_building_val,
)
from semantic_val.utils import utils

log = utils.get_logger(__name__)


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
        subsample_size: int = 30000,
        shuffle_train: bool = False,
    ):
        super().__init__()

        self.num_workers = num_workers

        self.lasfiles_dir = lasfiles_dir
        self.metadata_shapefile = metadata_shapefile
        self.datasplit_csv_filepath = metadata_shapefile.replace(".shp", ".csv")

        self.train_frac = train_frac
        self.subtile_width_meters = subtile_width_meters
        self.train_subtiles_by_tile = train_subtiles_by_tile
        self.subtile_overlap = subtile_overlap
        self.subsample_size = subsample_size
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # @property
    # def n_classes(self) -> int:
    #     return 2

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
        """
        Stratify train/val/test data if needed.
        Nota: Do not use it to assign state (self.x = y). This method is called only from a single GPU.
        """

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
                    method="random",
                ),
                ToTensor(),
                MakeCopyOfPosAndY(),
                FixedPointsPosXY(
                    self.subsample_size, replace=False, allow_duplicates=True
                ),
                MakeCopyOfSampledPos(),
                CustomNormalizeScale(),
                NormalizeFeatures(),
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
                MakeCopyOfPosAndY(),
                FixedPointsPosXY(
                    self.subsample_size, replace=False, allow_duplicates=True
                ),
                MakeCopyOfSampledPos(),
                CustomNormalizeScale(),
                NormalizeFeatures(),
            ]
        )

    def get_test_transforms(self) -> Compose:
        return self.get_val_transforms()

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.data_train, self.data_val, self.data_test.
        test_data = val data, because we only use all validation data after training.
        Test data can be used but only after final model is chosen.
        """

        df_split = pd.read_csv(self.datasplit_csv_filepath)

        df_split_train = df_split[df_split.split == "train"]
        df_split_train = df_split_train.sort_values("nb_bati", ascending=False)
        train_files = df_split_train.file_path.values.tolist()

        df_split_val = df_split[df_split.split == "val"]
        df_split_val = df_split_val.sort_values("nb_bati", ascending=False)
        val_files = df_split_val.file_path.values.tolist()
        test_files = val_files

        self.data_train = LidarTrainDataset(
            train_files,
            transform=self.get_train_transforms(),
            target_transform=MakeBuildingTargets(),
            subtile_width_meters=self.subtile_width_meters,
            train_subtiles_by_tile=self.train_subtiles_by_tile,
        )
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
            shuffle=False,
            sampler=TrainSampler(
                self.data_train.nb_files,
                self.train_subtiles_by_tile,
                shuffle=self.shuffle_train,
            ),
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )


class TrainSampler(Sampler[int]):
    """Custom sampler to draw multiple subtiles from a file in the same batch."""

    data_source: Sized

    def __init__(
        self, nb_files: int, train_subtiles_by_tile: int, shuffle: bool = False
    ) -> None:
        """:param data_source_size: Number of training LAS files."""
        self.nb_files = nb_files
        self.data_source_range = list(range(self.nb_files))
        self.train_subtiles_by_tile = train_subtiles_by_tile
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """Shuffle the files query indexes, and n-plicate them while keeping their new order."""
        if self.shuffle:
            random.shuffle(self.data_source_range)
        extended_range = [
            ([file_idx] * self.train_subtiles_by_tile)
            for file_idx in self.data_source_range
        ]
        return itertools.chain.from_iterable(extended_range)

    def __len__(self) -> int:
        return self.nb_files * self.train_subtiles_by_tile
