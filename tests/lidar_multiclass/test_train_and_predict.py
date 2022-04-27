import os.path as osp
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import pdal
import pytest
from lidar_multiclass.data.loading import LAS_SUBSET_FOR_TOY_DATASET
from lidar_multiclass.predict import predict
from lidar_multiclass.train import train
from tests.conftest import run_hydra_decorated_command
import laspy

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""


@pytest.mark.slow()
def test_FrenchLidar_default_training_fast_dev_run_as_command(
    isolated_toy_dataset_tmpdir,
):
    """Test running for 1 train, val and test batch."""

    command = [
        "run.py",
        "experiment=RandLaNet_base_run_FR",  # Use the defaults for French Lidar HD
        "logger=csv",  # disables comet logging
        f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
        "++trainer.fast_dev_run=true",  # Only one batch for train, val, and test.
    ]

    run_hydra_decorated_command(command)


@pytest.mark.slow()
def test_RandLaNet_overfitting(make_default_hydra_cfg, isolated_toy_dataset_tmpdir):
    with TemporaryDirectory() as tmpdir:

        hydra_overrides = make_hydra_overrides_for_training(
            isolated_toy_dataset_tmpdir, tmpdir
        )
        cfg = make_default_hydra_cfg(
            overrides=[
                "experiment=RandLaNetDebug"  # Use an experiment designe for overfitting a batch
            ]
            + hydra_overrides
        )
        train(cfg)
        # Not sure if version_0 is added by pytest or by lightning, but it is needed.
        metrics = pd.read_csv(osp.join(tmpdir, "csv", "version_0", "metrics.csv"))
        # Assert that there was a significative improvement i.e. the model learns.
        iou = metrics["train/iou_CLASS_building"].dropna()
        improvement = iou.iloc[-1] - iou.iloc[0]
        assert improvement >= 0.75


@pytest.mark.slow()
def test_PointNet_overfitting(make_default_hydra_cfg, isolated_toy_dataset_tmpdir):
    with TemporaryDirectory() as tmpdir:
        cfg = make_default_hydra_cfg(
            overrides=[
                "experiment=RandLaNetDebug",  # Use an experiment designe for overfitting a batch
                "model=point_net_model",  # but change model to PointNet
                "model.lr=0.001",  # need higher LR for PointNet
                "datamodule.batch_size=2",  # Smaller batch size for faster overfit
                # Define the task as a classification of all (1 and 2) vs. 6=building
                "datamodule.dataset_description.classification_preprocessing_dict={2:1}",
                f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
                "logger=csv",  # disables comet logging
                f"logger.csv.save_dir={tmpdir}",
            ]
        )
        train(cfg)
        # Not sure if version_0 is added by pytest or by lightning, but it is needed.
        metrics = pd.read_csv(osp.join(tmpdir, "csv", "version_0", "metrics.csv"))
        iou = metrics["train/loss_step"].dropna()
        first = iou.iloc[0]
        last = iou.iloc[-1]
        # Assert that loss was almost divided by two
        assert pytest.approx(last, 0.15) == first / 2
        # Note that for this toy dataset PointNet gets a null IoU for building class, which may be due
        # to an inability to catch buildings, which are a rare class


@pytest.mark.slow()
def test_RandLaNet_train_one_epoch_and_test(
    make_default_hydra_cfg, isolated_toy_dataset_tmpdir
):
    """Train for one epoch, and run test and predict functions using the trained model.

    Args:
        make_default_hydra_cfg (Callable): factory to instantiate hydra config
        isolated_toy_dataset_tmpdir (str): directory to toy dataset

    """
    with TemporaryDirectory() as tmpdir:
        hydra_overrides = make_hydra_overrides_for_training(
            isolated_toy_dataset_tmpdir, tmpdir
        )

        # Use an experiment designe for overfitting a batch
        cfg_one_epoch = make_default_hydra_cfg(
            overrides=[
                "experiment=RandLaNetDebug",
                "trainer.min_epochs=1",  # a single epoch
                "trainer.max_epochs=1",
            ]
            + hydra_overrides
        )
        trainer = train(cfg_one_epoch)

        # Use an experiment designed for testing on test set
        trained_model_path = trainer.checkpoint_callback.best_model_path
        cfg_test_using_trained_model = make_default_hydra_cfg(
            overrides=[
                "experiment=evaluate_test_data",
                f"model.ckpt_path={trained_model_path}",
            ]
            + hydra_overrides
        )
        train(cfg_test_using_trained_model)

        # predict
        cfg_predict_using_trained_model = make_default_hydra_cfg(
            overrides=[
                f"predict.resume_from_checkpoint={trained_model_path}",
                f"predict.src_las={LAS_SUBSET_FOR_TOY_DATASET}",
                f"predict.output_dir={tmpdir}",
                "predict.probas_to_save=[building,unclassified]",
            ]
            + hydra_overrides
        )
        output_las_path = predict(cfg_predict_using_trained_model)
        assert osp.isfile(output_las_path)

        # Check the format of the predicted las in terms of extra dimensions
        DIMS_ALWAYS_THERE = ["PredictedClassification", "entropy"]
        DIMS_CHOSEN_IN_CONFIG = [
            "building",
            "unclassified",
        ]
        check_las_contains_dims(
            output_las_path,
            dims_to_check=DIMS_ALWAYS_THERE + DIMS_CHOSEN_IN_CONFIG,
        )
        check_las_does_not_contains_dims(output_las_path, dims_to_check=["ground"])

        # check that predict does not change other dimensions
        check_las_invariance(LAS_SUBSET_FOR_TOY_DATASET, output_las_path)


def check_las_contains_dims(las1, dims_to_check=[]):
    a1 = pdal_read_las_array(las1)
    for d in dims_to_check:
        assert d in a1.dtype.fields.keys()


def check_las_does_not_contains_dims(las1, dims_to_check=[]):
    a1 = pdal_read_las_array(las1)
    for d in dims_to_check:
        assert d not in a1.dtype.fields.keys()


def pdal_read_las_array(in_f: str):
    """Read LAS as a named array.

    Args:
        in_f (str): input LAS path

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.
    """
    p1 = pdal.Pipeline() | pdal.Reader.las(filename=in_f)
    p1.execute()
    return p1.arrays[0]


def check_las_invariance(las1, las2):
    TOLERANCE = 0.0001

    a1 = pdal_read_las_array(las1)
    a2 = pdal_read_las_array(las2)
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert a1.shape == a2.shape  # no loss of points
    assert all(d in a2.dtype.fields.keys() for d in key_dims)  # key dims are here

    # order of points is allowed to change, so we assess a relaxed equality.
    for d in key_dims:
        assert pytest.approx(np.min(a2[d]), TOLERANCE) == np.min(a1[d])
        assert pytest.approx(np.max(a2[d]), TOLERANCE) == np.max(a1[d])
        assert pytest.approx(np.mean(a2[d]), TOLERANCE) == np.mean(a1[d])
        assert pytest.approx(np.sum(a2[d]), TOLERANCE) == np.sum(a1[d])


def make_hydra_overrides_for_training(isolated_toy_dataset_tmpdir, tmpdir):
    """Get list of overrides for hydra, the ones that are always needed when calling train/test."""

    return [
        "datamodule.batch_size=2",  # Smaller batch size for faster overfit
        f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
        "logger=csv",  # disables comet logging
        f"logger.csv.save_dir={tmpdir}",
        f"callbacks.model_checkpoint.dirpath={tmpdir}",
    ]
