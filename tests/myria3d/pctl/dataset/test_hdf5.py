"""Tests for myria3d.pctl.dataset.hdf5.create_hdf5.

The deterministic train transforms (target mapping, class dropping, grid sampling) can be
either baked once into the HDF5 file at creation time (``train_pre_transform``) or applied
on-the-fly at load time (``train_transform``). Both strategies must produce the exact same
train samples.
"""

import os.path as osp

import torch
from torch_geometric.transforms import GridSampling

from myria3d.pctl.dataset.hdf5 import HDF5Dataset
from myria3d.pctl.dataset.toy_dataset import (
    TOY_CLASSIFICATION_DICT,
    TOY_CLASSIFICATION_PREPROCESSING_DICT,
    TOY_EPSG,
    TOY_LAS_DATA,
)
from myria3d.pctl.transforms.compose import CustomCompose
from myria3d.pctl.transforms.transforms import DropPointsByClass, TargetTransform


def _deterministic_train_transforms():
    """Transforms that can be baked (train_pre_transform) or applied at load (train_transform)."""
    return CustomCompose(
        [
            TargetTransform(
                TOY_CLASSIFICATION_PREPROCESSING_DICT,
                TOY_CLASSIFICATION_DICT,
            ),
            DropPointsByClass(),
            GridSampling(0.25),
        ]
    )


def _make_hdf5_dataset(hdf5_path: str, bake: bool) -> HDF5Dataset:
    """Build a toy train-only HDF5 dataset, baking the transforms or applying them at load time.

    Args:
        hdf5_path: where to write the HDF5 file.
        bake: if True, pass the transforms as ``train_pre_transform`` (baked into the file);
            if False, store raw data and pass the same transforms as ``train_transform``.

    """
    transforms = _deterministic_train_transforms()
    return HDF5Dataset(
        hdf5_path,
        TOY_EPSG,
        las_paths_by_split_dict={"train": [TOY_LAS_DATA]},
        tile_width=110,
        subtile_width=50,
        train_pre_transform=transforms if bake else None,
        train_transform=None if bake else transforms,
        eval_transform=None,
        pre_filter=None,
    )


def test_baked_train_pre_transform_matches_unbaked(tmp_path):
    """create_hdf5 with a baked train_pre_transform yields the same train samples as storing
    raw data and applying the same transforms, unbaked, at load time."""
    baked = _make_hdf5_dataset(osp.join(str(tmp_path), "baked.hdf5"), bake=True)
    unbaked = _make_hdf5_dataset(osp.join(str(tmp_path), "unbaked.hdf5"), bake=False)

    baked_train = baked.traindata
    unbaked_train = unbaked.traindata

    # Same number of (non-empty) train samples, and the test is not vacuous.
    assert len(baked_train) == len(unbaked_train)
    assert len(baked_train) > 0

    for baked_sample, unbaked_sample in zip(baked_train, unbaked_train):
        assert baked_sample is not None
        assert unbaked_sample is not None
        # Same geometry/features/targets after the deterministic transforms.
        assert baked_sample.num_nodes == unbaked_sample.num_nodes
        assert torch.equal(baked_sample.y, unbaked_sample.y)
        assert torch.allclose(baked_sample.pos, unbaked_sample.pos)
        assert torch.allclose(baked_sample.x, unbaked_sample.x)
