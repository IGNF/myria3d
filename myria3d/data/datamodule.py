import os.path as osp
import glob
import numpy as np
from typing import Optional, List
from numbers import Number
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch_geometric.data.data import Data
from myria3d.utils import utils
from myria3d.data.transforms import CustomCompose


log = utils.get_logger(__name__)


class DataModule(LightningDataModule):
    """Datamodule to feed train and validation data to the model.

    At fit time (train+validate), data is loaded from a prepared dataset (see. loading.py).
    At test and inference time, data is loader from a raw point cloud directly for on-the-fly preparation.

    """

    def __init__(self, **kwargs):
        super().__init__()
        # paths
        self.prepared_data_dir = kwargs.get("prepared_data_dir")
        self.test_data_dir = kwargs.get("test_data_dir")
        # compute
        self.num_workers = kwargs.get("num_workers", 0)
        self.prefetch_factor = kwargs.get("prefetch_factor", 2)
        # data preparation
        self.subtile_width_meters = kwargs.get("subtile_width_meters", 50)
        self.subtile_overlap = kwargs.get("subtile_overlap", 0)
        self.batch_size = kwargs.get("batch_size", 32)
        self.augmentation_transforms = kwargs.get("augmentation_transforms", [])
        # segmentation task
        self.dataset_description = kwargs.get("dataset_description")
        self.classification_dict = self.dataset_description.get("classification_dict")
        self.classification_preprocessing_dict = self.dataset_description.get(
            "classification_preprocessing_dict"
        )
        self.load_las = self.dataset_description.get("load_las_func")
        # transforms
        t = kwargs.get("transforms")
        self.preparation_transforms = t.get("preparations_list")
        self.augmentation_transforms = t.get("augmentations_list")
        self.normalization_transforms = t.get("normalizations_list")

    def setup(self, stage: Optional[str] = None):
        """Loads data. Sets variables: self.data_train, self.data_val, self.data_test.
        :meta private:

        """
        if stage == "fit" or stage is None:
            self._set_train_data()
            self._set_val_data()

        if stage == "test" or stage is None:
            self._set_test_data()

    def _set_train_data(self):
        """Sets the train dataset from a directory."""
        files = glob.glob(
            osp.join(self.prepared_data_dir, "train", "**", "*.data"), recursive=True
        )
        self.train_data = LidarMapDataset(
            files, loading_function=torch.load, transform=self._get_train_transforms()
        )

    def _set_val_data(self):
        """Sets the validation dataset from a directory."""

        files = glob.glob(
            osp.join(self.prepared_data_dir, "val", "**", "*.data"), recursive=True
        )
        log.info(f"Validation on {len(files)} subtiles.")
        self.val_data = LidarMapDataset(
            files, loading_function=torch.load, transform=self._get_val_transforms()
        )

    def _set_test_data(self):
        """Sets the test dataset. User need to explicitely require the use of test set, which is kept out of experiment until the end."""

        files = glob.glob(osp.join(self.test_data_dir, "**", "*.las"), recursive=True)
        self.test_data = LidarIterableDataset(
            files,
            loading_function=self.load_las,
            transform=self._get_test_transforms(),
            subtile_width_meters=self.subtile_width_meters,
            subtile_overlap=self.subtile_overlap,
        )

    def _set_predict_data(self, files: List[str]):
        """Sets predict data from a single file. To be used in predict.py.

        NB: the single fgile should be in a list.

        """
        self.predict_data = LidarIterableDataset(
            files,
            loading_function=self.load_las,
            transform=self._get_predict_transforms(),
            subtile_width_meters=self.subtile_width_meters,
            subtile_overlap=self.subtile_overlap,
        )

    def train_dataloader(self):
        """Sets train dataloader."""
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        """Sets validation dataloader."""
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        """Sets test dataloader.

        The dataloader will produces batches of prepared subtiles from a single tile (point cloud).

        """
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,  # b/c terable dataloader
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        """Sets predict dataloader.

        The dataloader will produces batches of prepared subtiles from a single tile (point cloud).

        """
        return DataLoader(
            dataset=self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,  # b/c terable dataloader
            prefetch_factor=self.prefetch_factor,
        )

    def _get_train_transforms(self) -> CustomCompose:
        """Creates a transform composition for train phase."""
        return CustomCompose(
            self.preparation_transforms
            + self.augmentation_transforms
            + self.normalization_transforms
        )

    def _get_val_transforms(self) -> CustomCompose:
        """Creates a transform composition for val phase."""
        return CustomCompose(
            self.preparation_transforms + self.normalization_transforms
        )

    def _get_test_transforms(self) -> CustomCompose:
        """Creates a transform composition for test phase."""
        return self._get_val_transforms()

    def _get_predict_transforms(self) -> CustomCompose:
        """Creates a transform composition for predict phase."""
        return self._get_val_transforms()


class LidarMapDataset(Dataset):
    """A Dataset to load prepared data as produced via loading.py."""

    def __init__(
        self,
        files: List[str],
        loading_function=None,
        transform=None,
    ):
        self.files = files
        self.num_files = len(self.files)

        self.loading_function = loading_function
        self.transform = transform

    def __getitem__(self, idx):
        """Loads a subtile and transforms its features and targets."""
        filepath = self.files[idx]

        data = self.loading_function(filepath)
        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return self.num_files


class LidarIterableDataset(IterableDataset):
    """A Dataset to load a full point cloud, batch by batch."""

    def __init__(
        self,
        files,
        loading_function=None,
        transform=None,
        subtile_width_meters: Number = 50,
        subtile_overlap: Number = 0,
    ):
        self.files = files
        self.loading_function = loading_function
        self.transform = transform
        self.subtile_width_meters = subtile_width_meters
        self.subtile_overlap = subtile_overlap

    def yield_transformed_subtile_data(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""

        for idx, filepath in enumerate(self.files):
            log.info(f"Parsing file {idx+1}/{len(self.files)} [{filepath}]")
            tile_data = self.loading_function(filepath)
            centers = self.get_all_subtiles_xy_min_corner(tile_data)
            # TODO: change to process time function
            for xy_min_corner in centers:
                data = self.extract_subtile_from_tile_data(tile_data, xy_min_corner)
                if len(data.pos) < 50:
                    continue
                if self.transform:
                    data = self.transform(data)
                if data and (len(data.pos) > 50):
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
                start=low[0],
                stop=high[0] + 1,
                step=self.subtile_width_meters - self.subtile_overlap,
            )
            for y in np.arange(
                start=low[1],
                stop=high[1] + 1,
                step=self.subtile_width_meters - self.subtile_overlap,
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
        sub.idx_in_original_cloud = sub.idx_in_original_cloud[mask]
        return sub
