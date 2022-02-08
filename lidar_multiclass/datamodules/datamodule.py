import os.path as osp
import glob
import time
import numpy as np
from typing import Optional, List, AnyStr

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset
from torch_geometric.transforms import RandomFlip
from torch_geometric.data.data import Data
from torch_geometric.transforms.center import Center
from lidar_multiclass.utils import utils
from lidar_multiclass.datamodules.transforms import *

from lidar_multiclass.utils import utils

log = utils.get_logger(__name__)


class DataModule(LightningDataModule):
    """
    Datamdule to feed train and validation data to the model.
    We use a custome sampler during training to consecutively load several
    subtiles from a single tile, to reduce the I/O footprint.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # TODO: try to use save_hyperparameters to lightne this code.
        self.prepared_data_dir = kwargs.get("prepared_data_dir")

        self.num_workers = kwargs.get("num_workers", 0)

        self.subtile_width_meters = kwargs.get("subtile_width_meters", 50)
        self.subsample_size = kwargs.get("subsample_size", 12500)
        self.batch_size = kwargs.get("batch_size", 32)
        self.augment = kwargs.get("augment", True)

        self.dataset_description = kwargs.get("dataset_description")
        self.classification_dict = self.dataset_description.get("classification_dict")
        self.classification_preprocessing_dict = self.dataset_description.get(
            "classification_preprocessing_dict"
        )
        # By default, do not use the test set unless explicitely required by user.
        self.use_val_data_at_test_time = kwargs.get("use_val_data_at_test_time", True)

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.predict_data: Optional[Dataset] = None

        self.load_las = self.dataset_description.get("load_las_func")

    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: self.data_train, self.data_val, self.data_test.
        test_data = val data, because we only use all validation data after training.
        Test data can be used but only after final model is chosen.
        """
        self._set_all_transforms()

        if stage == "fit" or stage is None:
            self._set_train_data()
            self._set_val_data()

        if stage == "test" or stage is None:
            self._set_test_data()

    def _set_train_data(self):
        """Get the train dataset"""
        files = glob.glob(
            osp.join(self.prepared_data_dir, "train", "**", "*.data"), recursive=True
        )
        self.train_data = LidarMapDataset(
            files,
            loading_function=torch.load,
            transform=self._get_train_transforms(),
            target_transform=TargetTransform(
                self.classification_preprocessing_dict,
                self.classification_dict,
            ),
        )

    def _set_val_data(self):
        """Get the validation dataset"""
        files = glob.glob(
            osp.join(self.prepared_data_dir, "val", "**", "*.data"), recursive=True
        )
        log.info(f"Validation on {len(files)} subtiles.")
        self.val_data = LidarMapDataset(
            files,
            loading_function=torch.load,
            transform=self._get_val_transforms(),
            target_transform=TargetTransform(
                self.classification_preprocessing_dict,
                self.classification_dict,
            ),
        )

    def _set_test_data(self):
        """Get the test dataset. User need to explicitely require the use of test set, which is kept out of experiment until the end."""
        if self.use_val_data_at_test_time:
            self._set_val_data()
            self.test_data = self.val_data
            log.info(
                "Using validation data as test data. Use real test data with use_val_data_at_test_time=False at run time."
            )
            return
        files = glob.glob(
            osp.join(self.prepared_data_dir, "test", "**", "*.data"), recursive=True
        )
        self.test_data = LidarMapDataset(
            files,
            loading_function=torch.load,
            transform=self._get_test_transforms(),
            target_transform=TargetTransform(
                self.classification_preprocessing_dict, self.classification_dict
            ),
        )

    def _set_predict_data(self, files_to_infer_on: List[AnyStr]):
        """This is used in predict.py, with a single file in a list."""
        self.predict_data = LidarIterableDataset(
            files_to_infer_on,
            loading_function=self.load_las,
            transform=self._get_predict_transforms(),
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
            prefetch_factor=1,
        )

    def val_dataloader(self):
        """Get val dataloader."""
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=1,
        )

    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            prefetch_factor=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,  # b/c terable dataloader
            collate_fn=collate_fn,
            prefetch_factor=1,
        )

    def _set_all_transforms(self):
        """
        Set transforms that are shared between train/val-test.
        Called at initialization.
        """

        self.preparation = [
            EmptySubtileFilter(),
            ToTensor(),
            MakeCopyOfPos(),
            FixedPointsPosXY(self.subsample_size, replace=False, allow_duplicates=True),
            MakeCopyOfSampledPos(),
            Center(),
        ]
        self.augmentation = []
        if self.augment:
            self.augmentation = [RandomFlip(0, p=0.5), RandomFlip(1, p=0.5)]
        self.normalization = [NormalizePos(), StandardizeFeatures()]

    def _get_train_transforms(self) -> CustomCompose:
        """Create a transform composition for train phase."""
        return CustomCompose(self.preparation + self.augmentation + self.normalization)

    def _get_val_transforms(self) -> CustomCompose:
        """Create a transform composition for val phase."""
        return CustomCompose(self.preparation + self.normalization)

    def _get_test_transforms(self) -> CustomCompose:
        """Create a transform composition for test phase."""
        return self._get_val_transforms()

    def _get_predict_transforms(self) -> CustomCompose:
        """Create a transform composition for predict phase."""
        return self._get_val_transforms()


class LidarMapDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        loading_function=None,
        transform=None,
        target_transform=None,
    ):
        self.files = files
        self.num_files = len(self.files)

        self.loading_function = loading_function
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        """Load a subtile and apply the transforms specified in datamodule."""
        filepath = self.files[idx]

        data = self.loading_function(filepath)
        if self.transform:
            data = self.transform(data)
        if data is None:
            return None
        if self.target_transform:
            data = self.target_transform(data)

        return data

    def __len__(self):
        return self.num_files


class LidarIterableDataset(IterableDataset):
    def __init__(
        self,
        files,
        loading_function=None,
        transform=None,
        target_transform=None,
        subtile_width_meters: float = 50,
    ):
        self.files = files
        self.loading_function = loading_function
        self.transform = transform
        self.target_transform = target_transform
        self.subtile_width_meters = subtile_width_meters

    @utils.eval_time
    def yield_transformed_subtile_data(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""

        for idx, filepath in enumerate(self.files):
            log.info(f"Predicting for file {idx+1}/{len(self.files)} [{filepath}]")
            tile_data = self.loading_function(filepath)
            centers = self.get_all_subtiles_xy_min_corner(tile_data)
            # TODO: change to process time function
            ts = time.time()
            for xy_min_corner in centers:
                data = self.extract_subtile_from_tile_data(tile_data, xy_min_corner)
                if self.transform:
                    data = self.transform(data)
                if data is not None:
                    if self.target_transform:
                        data = self.target_transform(data)
                    yield data

    def __iter__(self):
        return self.yield_transformed_subtile_data()

    def get_all_subtiles_xy_min_corner(self, data: Data):
        """Get centers of square subtiles of specified width, assuming rectangular form of input cloud."""

        low = data.pos[:, :2].min(0)
        high = data.pos[:, :2].max(0)
        xy_min_corners = [
            np.array([x, y])
            for x in np.arange(
                start=low[0], stop=high[0] + 1, step=self.subtile_width_meters
            )
            for y in np.arange(
                start=low[1], stop=high[1] + 1, step=self.subtile_width_meters
            )
        ]
        # random.shuffle(centers)
        return xy_min_corners

    def extract_subtile_from_tile_data(self, data: Data, low_xy):
        """Extract the subset from xy_min_corner to xy_min_corner + self.subtile_width_meters

        Args:
            tile_data (Data): The full tile data.
            xy_min_corner (np.array): Coordonates of xy min corner of subtile to extract.
        """
        high_xy = low_xy + self.subtile_width_meters
        mask_x = (low_xy[0] <= data.pos[:, 0]) & (data.pos[:, 0] <= high_xy[0])
        mask_y = (low_xy[1] <= data.pos[:, 1]) & (data.pos[:, 1] <= high_xy[1])
        mask = mask_x & mask_y

        sub = data.clone()
        sub.pos = sub.pos[mask]
        sub.x = sub.x[mask]
        sub.y = sub.y[mask]
        return sub
