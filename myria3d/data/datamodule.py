import os.path as osp
import glob
from typing import Optional, List
from matplotlib import pyplot as plt
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm
from myria3d.data.loading import (
    MIN_NUM_NODES_PER_RECEPTIVE_FIELD,
    FrenchLidarDataSignature,
    LidarDataSignature
)
from myria3d.utils import utils
from torch_geometric.transforms import Compose


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
        self.data_signature = self.dataset_description.get(
            "data_signature", FrenchLidarDataSignature()
        )
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
        self.train_data = LidarMapDataset(files, transform=self._get_train_transforms())

    def _set_val_data(self):
        """Sets the validation dataset from a directory."""

        files = glob.glob(
            osp.join(self.prepared_data_dir, "val", "**", "*.data"), recursive=True
        )
        log.info(f"Validation on {len(files)} subtiles.")
        self.val_data = LidarMapDataset(files, transform=self._get_val_transforms())

    def _set_test_data(self):
        """Sets the test dataset."""

        files = glob.glob(osp.join(self.test_data_dir, "**", "*.las"), recursive=True)
        self.test_data = LidarIterableDataset(
            files,
            transform=self._get_test_transforms(),
            data_signature=self.data_signature,
        )

    def _set_predict_data(self, src_file: str):
        """Sets predict data from a single file. To be used in predict.py.

        NB: the single fgile should be in a list.

        """
        self.predict_data = LidarIterableDataset(
            [src_file],
            transform=self._get_predict_transforms(),
            data_signature=self.data_signature,
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
            num_workers=1,  # b/c iterable dataset
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self):
        """Sets predict dataloader.

        The dataloader will produces batches of prepared subtiles from a single tile (point cloud).

        """
        return DataLoader(
            dataset=self.predict_data,
            batch_size=self.batch_size,
            num_workers=1,  # b/c iterable dataset
            prefetch_factor=self.prefetch_factor,
        )

    def _get_train_transforms(self) -> Compose:
        """Creates a transform composition for train phase."""
        return Compose(
            self.preparation_transforms
            + self.augmentation_transforms
            + self.normalization_transforms
        )

    def _get_val_transforms(self) -> Compose:
        """Creates a transform composition for val phase."""
        return Compose(self.preparation_transforms + self.normalization_transforms)

    def _get_test_transforms(self) -> Compose:
        """Creates a transform composition for test phase."""
        return self._get_val_transforms()

    def _get_predict_transforms(self) -> Compose:
        """Creates a transform composition for predict phase."""
        return self._get_val_transforms()

    def _visualize_graph(self, data, color=None):
        """Helpful function to plot the graph, colored by class if not specified.

        Args:
            data (Data): data, usually post-transform to show their effect.
            color (Tensor, optional): array with which to color the graph.
        """

        # creating an empty canvas

        plt.figure(figsize=(20, 20))

        # defining the axes with the projection
        # as 3D so as to plot 3D graphs
        ax = plt.axes(projection="3d")
        ax.set_xlim([-self.subtile_width_meters / 2, self.subtile_width_meters / 2])
        ax.set_ylim([-self.subtile_width_meters / 2, self.subtile_width_meters / 2])
        ax.set_zlim([0, 25])

        # plotting a scatter plot with X-coordinate,
        # Y-coordinate and Z-coordinate respectively
        # and defining the points color as cividis
        # and defining c as z which basically is a
        # defination of 2D array in which rows are RGB
        # or RGBA
        if not color:
            color = data.y
        ax.scatter3D(
            data.pos[:, 0],
            data.pos[:, 1],
            data.pos[:, 2],
            c=color,
            cmap="cividis",
        )

        # Showing the above plot
        plt.show()


class LidarMapDataset(Dataset):
    """A Dataset to load prepared data produced via loading.py."""

    def __init__(self, files: List[str], transform=None):
        self.files = files
        self.num_files = len(self.files)
        self.transform = transform

    def __getitem__(self, idx):
        """Loads a subtile and transforms its features and targets."""
        filepath = self.files[idx]

        data = torch.load(filepath)
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
        transform=None,
        data_signature: LidarDataSignature = FrenchLidarDataSignature,
    ):
        self.files = files
        self.transform = transform
        self.data_signature = data_signature

    def __iter__(self):
        return self.get_iterator()

    def get_iterator(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""
        for f in tqdm(self.files):
            log.info(f"Parsing file: {f}")
            for data_sample in self.data_signature.split_cloud_into_receptive_fields(f):
                if self.transform:
                    data_sample = self.transform(data_sample)
                if data_sample and (
                    len(data_sample.pos) >= MIN_NUM_NODES_PER_RECEPTIVE_FIELD
                ):
                    yield data_sample
