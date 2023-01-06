import os.path as osp
from typing import List

import numpy as np
import pytest

from myria3d.pctl.dataset.toy_dataset import TOY_LAS_DATA
from myria3d.pctl.dataset.utils import pdal_read_las_array
from myria3d.predict import predict
from myria3d.train import train
from tests.conftest import make_default_hydra_cfg, run_hydra_decorated_command
from tests.runif import RunIf

"""
Sanity checks to make sure the model train/val/predict/test logics do not crash.
"""


@pytest.fixture(scope="session")
def one_epoch_trained_RandLaNet_checkpoint(toy_dataset_hdf5_path, tmpdir_factory):
    """Train a RandLaNet model for one epoch, in order to run it in different other tests.

    Args:
        toy_dataset_hdf5_path (str): path to toy dataset as created by fixture.
        tmpdir_factory (fixture): factory to create a session level tempdir.

    Returns:
        str: path to trained model checkpoint, which persists for the whole pytest session.

    """
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        toy_dataset_hdf5_path, tmpdir
    )
    cfg_one_epoch = make_default_hydra_cfg(
        overrides=["experiment=RandLaNetDebug"] + tmp_paths_overrides
    )
    trainer = train(cfg_one_epoch)
    return trainer.checkpoint_callback.best_model_path


@RunIf(min_gpus=1)
def test_FrenchLidar_RandLaNetDebug_with_gpu(toy_dataset_hdf5_path, tmpdir_factory):
    """Train a RandLaNet model for one epoch using GPU. XFail is no GPU available.

    Args:
        toy_dataset_hdf5_path (str): path to isolated toy dataset as created by fixture.
        tmpdir_factory (fixture): factory to create a session-level tempdir.

    """
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        toy_dataset_hdf5_path, tmpdir
    )
    # We will always use the first GPU id for tests, because it always exists if there are some GPUs.
    # Attention to concurrency with other processes using the GPU when running tests.
    gpu_id = 0
    cfg_one_epoch = make_default_hydra_cfg(
        overrides=["experiment=RandLaNetDebug", f"trainer.gpus=[{gpu_id}]"]
        + tmp_paths_overrides
    )
    train(cfg_one_epoch)


def test_predict_as_command(one_epoch_trained_RandLaNet_checkpoint, tmpdir):
    """Test running inference by CLI for toy LAS.

    Args:
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    # Hydra changes CWD, and therefore absolute paths are preferred
    abs_path_to_toy_LAS = osp.abspath(TOY_LAS_DATA)
    command = [
        "run.py",
        f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
        f"predict.src_las={abs_path_to_toy_LAS}",
        f"predict.output_dir={tmpdir}",
        "+predict.probas_to_save=[building,unclassified]",
        "task.task_name=predict",
    ]
    run_hydra_decorated_command(command)


def test_RandLaNet_predict_with_invariance_checks(
    one_epoch_trained_RandLaNet_checkpoint, tmpdir
):
    """Train a model for one epoch, and run test and predict functions using the trained model.

    Args:
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        "placeholder_because_no_need_for_a_dataset_here", tmpdir
    )
    # Run prediction
    cfg_predict_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=predict",
            f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
            f"predict.src_las={TOY_LAS_DATA}",
            f"predict.output_dir={tmpdir}",
            "+predict.interpolator.interpolation_k=predict.interpolation_k",
            "+predict.interpolator.probas_to_save=[building,unclassified]",
        ]
        + tmp_paths_overrides
    )
    path_to_output_las = predict(cfg_predict_using_trained_model)

    # Check that predict function generates a predicted LAS
    assert osp.isfile(path_to_output_las)

    # Check the format of the predicted las in terms of extra dimensions
    DIMS_ALWAYS_THERE = ["confidence", "entropy"]
    DIMS_CHOSEN_IN_CONFIG = ["building", "unclassified"]
    check_las_contains_dims(
        path_to_output_las,
        dims_to_check=DIMS_ALWAYS_THERE + DIMS_CHOSEN_IN_CONFIG,
    )
    DIMS_NOT_THERE = ["ground"]
    check_las_does_not_contains_dims(path_to_output_las, dims_to_check=DIMS_NOT_THERE)

    # check that predict does not change other dimensions
    check_las_invariance(TOY_LAS_DATA, path_to_output_las)


def test_run_test_with_trained_model_on_toy_dataset_on_cpu(
    one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir
):
    _run_test_right_after_training(
        one_epoch_trained_RandLaNet_checkpoint,
        toy_dataset_hdf5_path,
        tmpdir,
        "null",
    )


@RunIf(min_gpus=1)
def test_run_test_with_trained_model_on_toy_dataset_on_gpu(
    one_epoch_trained_RandLaNet_checkpoint, toy_dataset_hdf5_path, tmpdir
):
    _run_test_right_after_training(
        one_epoch_trained_RandLaNet_checkpoint,
        toy_dataset_hdf5_path,
        tmpdir,
        "[0]",
    )


def _run_test_right_after_training(
    one_epoch_trained_RandLaNet_checkpoint,
    toy_dataset_hdf5_path,
    tmpdir,
    trainer_gpus,
):
    """Run test using the model that was just trained for one epoch.

    Args:
        toy_dataset_hdf5_path (fixture -> str): path to toy dataset
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    # Run testing on toy testset with trainer.test(...)
    # function's name is train, but under the hood and thanks to configuration,
    # trainer.test(...) is called.
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        toy_dataset_hdf5_path, tmpdir
    )
    cfg_test_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=test",  # sets task.task_name to "test"
            f"model.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
            f"trainer.gpus={trainer_gpus}",
        ]
        + tmp_paths_overrides
    )
    train(cfg_test_using_trained_model)


def check_las_contains_dims(las_path: str, dims_to_check: List[str] = []):
    """Utility: check that LAS contains some dimensions.


    Args:
        las_path (str): path to LAS file.
        dims_to_check (List[str], optional): list of dimensions expected to be there. Defaults to [].

    """
    a1 = pdal_read_las_array(las_path)
    for dim in dims_to_check:
        assert dim in a1.dtype.fields.keys()


def check_las_does_not_contains_dims(las_path, dims_to_check=[]):
    """Utility: check that LAS does NOT contain some dimensions.


    Args:
        las_path (str): path to LAS file.
        dims_to_check (List[str], optional): list of dimensions expected not to be there. Defaults to [].

    """
    a1 = pdal_read_las_array(las_path)
    for dim in dims_to_check:
        assert dim not in a1.dtype.fields.keys()


def check_las_invariance(las_path_1: str, las_path_2: str):
    """Check that key dimensions are equal between two LAS files

    Args:
        las_path_1 (str): path to first LAS file.
        las_path_2 (str): path to second LAS file.

    """
    a1 = pdal_read_las_array(las_path_1)
    a2 = pdal_read_las_array(las_path_2)
    key_dims = ["X", "Y", "Z", "Infrared", "Red", "Blue", "Green", "Intensity"]
    assert a1.shape == a2.shape  # no loss of points
    assert all(d in a2.dtype.fields.keys() for d in key_dims)  # key dims are here

    # order of points is allowed to change, so we assess a relaxed equality.
    rel_tolerance = 0.0001
    for d in key_dims:
        assert pytest.approx(np.min(a2[d]), rel_tolerance) == np.min(a1[d])
        assert pytest.approx(np.max(a2[d]), rel_tolerance) == np.max(a1[d])
        assert pytest.approx(np.mean(a2[d]), rel_tolerance) == np.mean(a1[d])
        assert pytest.approx(np.sum(a2[d]), rel_tolerance) == np.sum(a1[d])


def _make_list_of_necesary_hydra_overrides_with_tmp_paths(
    toy_dataset_hdf5_path: str, tmpdir: str
):
    """Get list of overrides for hydra, the ones that are always needed when calling train/test.

    Args:
        toy_dataset_hdf5_path (str): path to directory to dataset.
        tmpdir (str): path to temporary directory.

    """

    return [
        f"datamodule.hdf5_file_path={toy_dataset_hdf5_path}",
        "logger=csv",  # disables comet logging
        f"logger.csv.save_dir={tmpdir}",
        f"callbacks.model_checkpoint.dirpath={tmpdir}",
    ]
