import numpy as np

from myria3d.pctl.dataset.toy_dataset import TOY_LAS_DATA
from myria3d.pctl.dataset.utils import split_cloud_into_samples


def test_test_get():
    """Check that default sample indices are a numpy array with size > 0"""
    for sample_idx, sample_points in split_cloud_into_samples(
        TOY_LAS_DATA, tile_width=1000, subtile_width=50, epsg="2154", subtile_overlap=0
    ):
        assert isinstance(sample_idx, np.ndarray)
        assert sample_idx.size > 0


def test_split_cloud_into_samples_single_point_batch():
    """Check that sample indices are a numpy array with size > 0 when we have small batches with 1 point only"""
    single_point_batch_found = False
    for sample_idx, _ in split_cloud_into_samples(
        TOY_LAS_DATA, tile_width=1000, subtile_width=1, epsg="2154", subtile_overlap=0
    ):
        if len(sample_idx) == 1:
            single_point_batch_found = True
            assert isinstance(sample_idx, np.ndarray)
            assert sample_idx.size > 0
            break

    assert (
        single_point_batch_found
    ), "Test is not meaningfull as we could not isolate any batch with one point only"
