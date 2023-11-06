from numbers import Number
from typing import Callable, Dict, List, Optional

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from pytorch_lightning import LightningDataModule
from torch_geometric.data import Data

from myria3d.pctl.dataloader.dataloader import GeometricNoneProofDataloader
from myria3d.pctl.dataset.list import ListDataset
from myria3d.pctl.transforms.compose import CustomCompose
from myria3d.pctl.dataset.iterable import InferenceDataset
from myria3d.pctl.dataset.utils import (
    get_las_paths_by_split_dict,
    pre_filter_below_n_points,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)

TRANSFORMS_LIST = List[Callable]


class ListDataModule(LightningDataModule):
    """Datamodule to feed train and validation data to the model, already prepared."""

    def __init__(
        self,
        data_dir: str,
        split_csv_path: str,
        points_pre_transform: Optional[Callable[[ArrayLike], Data]] = None,
        pre_filter: Optional[Callable[[Data], bool]] = pre_filter_below_n_points,
        tile_width: int = 1000,
        subtile_width: Number = 50,
        subtile_overlap_train: Number = 0,
        subtile_overlap_predict: Number = 0,
        batch_size: int = 12,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        transforms: Optional[Dict[str, TRANSFORMS_LIST]] = None,
        **kwargs,
    ):
        self.split_csv_path = split_csv_path
        self.data_dir = data_dir
        self.las_paths_by_split_dict = {}  # Will be set from split_csv

        self.points_pre_transform = points_pre_transform
        self.pre_filter = pre_filter

        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap_train = subtile_overlap_train
        self.subtile_overlap_predict = subtile_overlap_predict

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.test_on_eval = kwargs.get("test_on_eval", False)

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        t = transforms
        self.preparation_train_transform: TRANSFORMS_LIST = t.get("preparations_train_list", [])
        self.preparation_eval_transform: TRANSFORMS_LIST = t.get("preparations_eval_list", [])
        self.preparation_predict_transform: TRANSFORMS_LIST = t.get(
            "preparations_predict_list", []
        )
        self.augmentation_transform: TRANSFORMS_LIST = t.get("augmentations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

    @property
    def train_transform(self) -> CustomCompose:
        return CustomCompose(
            self.preparation_train_transform
            + self.normalization_transform
            + self.augmentation_transform
        )

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
                las_paths_by_split_dict = get_las_paths_by_split_dict(
                    self.data_dir, self.split_csv_path
                )
            else:
                log.warning(
                    "cfg.data_dir and cfg.split_csv_path are both null. Precomputed HDF5 dataset is used."
                )
                las_paths_by_split_dict = None
        # Create the dataset in prepare_data, so that it is done one a single GPU.
        self.las_paths_by_split_dict = las_paths_by_split_dict

    @property
    def train_dataset(self) -> ListDataset:
        if self._train_dataset:
            return self._train_dataset

        self._train_dataset = ListDataset(
            las_paths_by_split_dict=self.las_paths_by_split_dict,
            split="train",
            points_pre_transform=self.points_pre_transform,
            subtile_overlap_train=self.subtile_overlap_train,
            pre_filter=self.pre_filter,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        return self._train_dataset

    @property
    def val_dataset(self) -> ListDataset:
        if self._val_dataset:
            return self._val_dataset

        self._val_dataset = ListDataset(
            las_paths_by_split_dict=self.las_paths_by_split_dict,
            split="val",
            points_pre_transform=self.points_pre_transform,
            subtile_overlap_train=self.subtile_overlap_train,
            pre_filter=self.pre_filter,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        return self._val_dataset

    @property
    def test_dataset(self) -> ListDataset:
        if self._test_dataset:
            return self._test_dataset

        self._test_dataset = ListDataset(
            las_paths_by_split_dict=self.las_paths_by_split_dict,
            split="test",
            points_pre_transform=self.points_pre_transform,
            subtile_overlap_train=self.subtile_overlap_train,
            pre_filter=self.pre_filter,
            train_transform=self.train_transform,
            eval_transform=self.eval_transform,
        )
        return self._test_dataset

    def train_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        if self.test_on_eval:
            return self.val_dataloader()
        return GeometricNoneProofDataloader(
            dataset=self.test_dataset,
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
            subtile_overlap=self.subtile_overlap_predict,
        )

    def predict_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=1,  # always 1 because this is an iterable dataset
            prefetch_factor=self.prefetch_factor,
        )
