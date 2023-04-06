from numbers import Number
from typing import Callable, Dict, List, Optional

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data

from myria3d.pctl.dataloader.dataloader import GeometricNoneProofDataloader
from myria3d.pctl.transforms.compose import CustomCompose
from myria3d.pctl.dataset.hdf5 import HDF5Dataset
from myria3d.pctl.dataset.iterable import InferenceDataset
from myria3d.pctl.dataset.utils import (
    SHAPE_TYPE,
    get_las_paths_by_split_dict,
    pre_filter_below_n_points,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)

TRANSFORMS_LIST = List[Callable]


class HDF5LidarDataModule(LightningDataModule):
    """Datamodule to feed train and validation data to the model."""

    def __init__(
        self,
        data_dir: str,
        split_csv_path: str,
        hdf5_file_path: str,
        points_pre_transform: Optional[Callable[[ArrayLike], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
        tile_width: Number = 1000,
        subtile_width: Number = 50,
        subtile_shape: SHAPE_TYPE = "square",
        subtile_overlap_train: Number = 0,
        subtile_overlap_predict: Number = 0,
        batch_size: int = 12,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        transforms: Optional[Dict[str, TRANSFORMS_LIST]] = None,
    ):
        self.split_csv_path = split_csv_path
        self.data_dir = data_dir
        self.hdf5_file_path = hdf5_file_path
        self._dataset = None  # will be set by self.dataset property
        self.las_paths_by_split_dict = {}  # Will be set from split_csv

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_shape = subtile_shape
        self.subtile_overlap_train = subtile_overlap_train
        self.subtile_overlap_predict = subtile_overlap_predict

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        t = transforms
        self.preparation_train_transform: TRANSFORMS_LIST = t.get("preparations_train_list", [])
        self.preparation_eval_transform: TRANSFORMS_LIST = t.get("preparations_eval_list", [])
        self.preparation_predict_transform: TRANSFORMS_LIST = t.get("preparations_predict_list", [])
        self.augmentation_transform: TRANSFORMS_LIST = t.get("augmentations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

    @property
    def train_transform(self) -> CustomCompose:
        return CustomCompose(self.preparation_train_transform + self.normalization_transform + self.augmentation_transform)

    @property
    def eval_transform(self) -> CustomCompose:
        return CustomCompose(self.preparation_eval_transform + self.normalization_transform)

    @property
    def predict_transform(self) -> CustomCompose:
        return CustomCompose(self.preparation_predict_transform + self.normalization_transform)

    def prepare_data(self, stage: Optional[str] = None):
        """Prepare dataset containing train, val, test data."""

        if stage in ["fit", "test"] or stage is None:
            if self.split_csv_path and self.data_dir:
                las_paths_by_split_dict = get_las_paths_by_split_dict(self.data_dir, self.split_csv_path)
            else:
                log.warning("cfg.data_dir and cfg.split_csv_path are both null. Precomputed HDF5 dataset is used.")
                las_paths_by_split_dict = None
        # Create the dataset in prepare_data, so that it is done one a single GPU.
        self.las_paths_by_split_dict = las_paths_by_split_dict
        self.dataset

    # TODO: not needed ?
    def setup(self, stage: Optional[str] = None) -> None:
        """Instantiate the (already prepared) dataset (called on all GPUs)."""
        self.dataset

    @property
    def dataset(self) -> HDF5Dataset:
        """Abstraction to ease HDF5 dataset instantiation.

        Args:
            las_paths_by_split_dict (LAS_PATHS_BY_SPLIT_DICT_TYPE, optional): Maps split (val/train/test) to file path.
                If specified, the hdf5 file is created at dataset initialization time.
                Otherwise,a precomputed HDF5 file is used directly without I/O to the HDF5 file.
                This is usefule for multi-GPU training, where data creation is performed in prepare_data method, and the dataset
                is then loaded again in each GPU in setup method.
                Defaults to None.

        Returns:
            HDF5Dataset: the dataset with train, val, and test data.

        """
        if self._dataset:
            return self._dataset

        self._dataset = HDF5Dataset(
            self.hdf5_file_path,
            las_paths_by_split_dict=self.las_paths_by_split_dict,
            points_pre_transform=self.points_pre_transform,
            tile_width=self.tile_width,
            subtile_width=self.subtile_width,
            subtile_overlap_train=self.subtile_overlap_train,
            subtile_shape=self.subtile_shape,
            pre_filter=self.pre_filter,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        return self._dataset

    def train_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.dataset.traindata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.dataset.valdata,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.dataset.testdata,
            batch_size=self.batch_size,
            num_workers=1,  # b/c iterable dataset
            prefetch_factor=self.prefetch_factor,
        )

    def _set_predict_data(self, las_file_to_predict):
        self.predict_dataset = InferenceDataset(
            las_file_to_predict,
            points_pre_transform=self.points_pre_transform,
            pre_filter=self.pre_filter,
            transform=self.predict_transform,
            tile_width=self.tile_width,
            subtile_width=self.subtile_width,
            subtile_shape=self.subtile_shape,
            subtile_overlap=self.subtile_overlap_predict,
        )

    def predict_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=1,  # always 1 because this is an iterable dataset
            prefetch_factor=self.prefetch_factor,
        )

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
        ax.set_xlim([-self.subtile_width / 2, self.subtile_width / 2])
        ax.set_ylim([-self.subtile_width / 2, self.subtile_width / 2])
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
