from numbers import Number
from typing import Callable, Dict, Optional, List
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
import pandas as pd
from myria3d.pctl.dataloader.dataloader import GeometricNoneProofDataloader
from myria3d.utils import utils
from pytorch_lightning import LightningDataModule
from torch_geometric.transforms import Compose
from torch_geometric.data import Data

from myria3d.pctl.dataset.iterable import InferenceDataset
from myria3d.pctl.dataset.utils import (
    SHAPE_TYPE,
    find_file_in_dir,
    pre_filter_below_n_points,
)
from myria3d.pctl.dataset.hdf5 import HDF5Dataset

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
        self.preparation_train_transform: TRANSFORMS_LIST = t.get(
            "preparations_train_list", []
        )
        self.preparation_eval_transform: TRANSFORMS_LIST = t.get(
            "preparations_eval_list", []
        )
        self.preparation_predict_transform: TRANSFORMS_LIST = t.get(
            "preparations_predict_list", []
        )

        self.augmentation_transform: TRANSFORMS_LIST = t.get("augmentations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

    @property
    def train_transform(self) -> Compose:
        return Compose(
            self.preparation_train_transform
            + self.normalization_transform
            + self.augmentation_transform
        )

    @property
    def eval_transform(self) -> Compose:
        return Compose(self.preparation_eval_transform + self.normalization_transform)

    @property
    def predict_transform(self) -> Compose:
        return Compose(
            self.preparation_predict_transform + self.normalization_transform
        )

    def prepare_data(self, stage: Optional[str] = None):
        """Prepare dataset containing train, val, test data."""
        if stage in ["fit", "test"] or stage is None:
            las_files_by_split = None
            if self.split_csv_path and self.data_dir:
                las_files_by_split = {}
                split_df = pd.read_csv(self.split_csv_path)
                for phase in ["train", "val", "test"]:
                    basenames = split_df[split_df.split == phase].basename.tolist()
                    las_files_by_split[phase] = [
                        find_file_in_dir(self.data_dir, b) for b in basenames
                    ]
                if not len(las_files_by_split):
                    raise FileNotFoundError(
                        (
                            f"No basename found while parsing directory {self.data_dir}"
                            f"using {self.split_csv_path} as split CSV."
                        )
                    )
            else:
                log.warning(
                    "cfg.data_dir and cfg.split_csv_path are both null. Precomputed HDF5 dataset is used."
                )
            self.dataset = HDF5Dataset(
                self.hdf5_file_path,
                las_files_by_split=las_files_by_split,
                points_pre_transform=self.points_pre_transform,
                tile_width=self.tile_width,
                subtile_width=self.subtile_width,
                subtile_overlap_train=self.subtile_overlap_train,
                subtile_shape=self.subtile_shape,
                pre_filter=self.pre_filter,
                train_transform=self.train_transform,
                eval_transform=self.eval_transform,
            )

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
