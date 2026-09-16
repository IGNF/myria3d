"""Datamodule for Pointcept-preprocessed downstream linear-probing datasets."""

from numbers import Number
from typing import Callable, Dict, List, Optional

import hydra
from pytorch_lightning import LightningDataModule

from myria3d.pctl.dataloader.dataloader import GeometricNoneProofDataloader
from myria3d.pctl.dataset.downstream.base import DownstreamNpyDataset
from myria3d.pctl.dataset.utils import pre_filter_below_n_points
from myria3d.pctl.transforms.compose import CustomCompose
from myria3d.utils import utils

log = utils.get_logger(__name__)

TRANSFORMS_LIST = List[Callable]


class DownstreamNpyDatamodule(LightningDataModule):
    """Datamodule reading a Pointcept-preprocessed downstream dataset (DALES/H3D/ECLAIR)
    for linear probing.

    Much simpler than `PointceptNpyDatamodule`: no CSV-manifest, iter-limited-sampler,
    or multitask-task-config plumbing -- just train/val/test directory listings routed
    through `dataset_target` (a `DownstreamNpyDataset` subclass dotted path, e.g.
    `myria3d.pctl.dataset.downstream.dales.DalesDataset`).

    DALES has no val split upstream: point `val_dir` at the same folder as `test_dir`
    (see readme_linear_probing.md) -- this is a deliberate, documented convention, not
    a silently-mislabeled held-out set.
    """

    def __init__(
        self,
        data_root: str,
        dataset_target: str,
        train_dir: str = "train",
        val_dir: str = "val",
        test_dir: str = "test",
        tile_width: Number = 50,
        subtile_width: Number = 50,
        subtile_overlap: Number = 0,
        pre_filter: Optional[Callable] = pre_filter_below_n_points,
        batch_size: int = 12,
        num_workers: int = 1,
        prefetch_factor: int = 2,
        transforms: Optional[Dict[str, TRANSFORMS_LIST]] = None,
        dataset_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.dataset_class = hydra.utils.get_class(dataset_target)
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.tile_width = tile_width
        self.subtile_width = subtile_width
        self.subtile_overlap = subtile_overlap
        self.pre_filter = pre_filter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dataset_kwargs = dict(dataset_kwargs or {})

        t = transforms or {}
        self.preparation_train_transform: TRANSFORMS_LIST = t.get("preparations_train_list", [])
        self.preparation_eval_transform: TRANSFORMS_LIST = t.get("preparations_eval_list", [])
        self.augmentation_transform: TRANSFORMS_LIST = t.get("augmentations_list", [])
        self.normalization_transform: TRANSFORMS_LIST = t.get("normalizations_list", [])

        self._train_dataset: Optional[DownstreamNpyDataset] = None
        self._val_dataset: Optional[DownstreamNpyDataset] = None
        self._test_dataset: Optional[DownstreamNpyDataset] = None

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

    def _build_dataset(self, split_dir: str, is_eval: bool) -> DownstreamNpyDataset:
        return self.dataset_class(
            data_root=self.data_root,
            split_dir=split_dir,
            is_eval=is_eval,
            tile_width=self.tile_width,
            subtile_width=self.subtile_width,
            subtile_overlap=self.subtile_overlap,
            pre_filter=self.pre_filter,
            transform=self.eval_transform if is_eval else self.train_transform,
            **self.dataset_kwargs,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset
        self.val_dataset
        self.test_dataset

    @property
    def train_dataset(self) -> DownstreamNpyDataset:
        if self._train_dataset is None:
            self._train_dataset = self._build_dataset(self.train_dir, is_eval=False)
        return self._train_dataset

    @property
    def val_dataset(self) -> DownstreamNpyDataset:
        if self._val_dataset is None:
            self._val_dataset = self._build_dataset(self.val_dir, is_eval=True)
        return self._val_dataset

    @property
    def test_dataset(self) -> DownstreamNpyDataset:
        if self._test_dataset is None:
            self._test_dataset = self._build_dataset(self.test_dir, is_eval=True)
        return self._test_dataset

    def train_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            shuffle=True,
        )

    def val_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return GeometricNoneProofDataloader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
