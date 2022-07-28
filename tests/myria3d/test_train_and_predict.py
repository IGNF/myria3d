import os.path as osp
import numpy as np
import pandas as pd
import pdal
import pytest
from typing import List

from myria3d.data.loading import LAS_SUBSET_FOR_TOY_DATASET
from myria3d.predict import predict
from myria3d.train import train
from tests.conftest import (
    TRAINED_MODEL_PATH,
    make_default_hydra_cfg,
    run_hydra_decorated_command,
)
from tests.runif import RunIf


"""
Sanity checks to make sure the model train/val/predict/test logics do not crash.
"""


def test_FrenchLidar_default_training_on_one_batch_as_command(
    isolated_toy_dataset_tmpdir,
):
    """Test running by CLI for 1 train, val and test batch of a toy dataset.

    Args:
        isolated_toy_dataset_tmpdir (fixture -> str): directory to toy dataset

    """

    command = [
        "run.py",
        "experiment=RandLaNet_base_run_FR",  # Use the defaults for French Lidar HD
        "logger=csv",  # disables comet logging
        f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
        "++trainer.fast_dev_run=1",  # Only one batch for train, val, test, predict
    ]

    run_hydra_decorated_command(command)


def test_predict_as_command(one_epoch_trained_RandLaNet_checkpoint, tmpdir):
    """Test running inference by CLI for toy LAS.

    Args:
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """
    # Hydra changes CWD, and therefore absolute paths are preferred
    abs_path_to_toy_LAS = osp.abspath(LAS_SUBSET_FOR_TOY_DATASET)
    command = [
        "run.py",
        "experiment=predict",
        # "task.task_name=predict",
        f"predict.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
        f"predict.src_las={abs_path_to_toy_LAS}",
        f"predict.output_dir={tmpdir}",
        "predict.probas_to_save=[building,unclassified]",
        # We set this because pytest fails to use PREPARED_DATA_DIR given via monkeypatch
        "datamodule.prepared_data_dir=placeholder",
    ]

    run_hydra_decorated_command(command)


@pytest.mark.slow()
def test_RandLaNet_overfitting(isolated_toy_dataset_tmpdir, tmpdir):
    """Check ability to overfit with RandLa-Net.

    Check that overfitting a single batch from a toy dataset, for 30 epochs, results
    in a significant improvement of building IoU.

    Args:
        isolated_toy_dataset_tmpdir (fixture -> str): directory to toy dataset
        tmpdir (fixture -> str): temporary directory.

    """

    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_toy_dataset_tmpdir, tmpdir
    )
    cfg = make_default_hydra_cfg(
        overrides=[
            "experiment=RandLaNetDebug",  # Use an experiment designe for overfitting a batch
            "datamodule.batch_size=2",  # Smaller batch size for faster overfit
        ]
        + tmp_paths_overrides
    )
    train(cfg)
    # Not sure if version_0 is added by pytest or by lightning, but it is needed.
    metrics = _get_metrics_df_from_tmpdir(tmpdir)
    # Assert that there was a significative improvement i.e. the model learns.
    iou = metrics["train/iou_CLASS_building"].dropna()
    improvement = iou.iloc[-1] - iou.iloc[0]
    assert improvement >= 0.6


@pytest.mark.slow()
def test_PointNet2_overfitting(isolated_toy_dataset_tmpdir, tmpdir):
    """Check ability to overfit with PointNet2.

    Check that overfitting a single batch from a toy dataset, for 30 epochs, results
    in significanly lower training loss.

    Args:
        isolated_toy_dataset_tmpdir (fixture -> str): directory to toy dataset
        tmpdir (fixture -> str): temporary directory.

    """
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_toy_dataset_tmpdir, tmpdir
    )
    cfg = make_default_hydra_cfg(
        overrides=tmp_paths_overrides
        + [
            "experiment=PointNet2Debug",  # Use an experiment designed for overfitting a batch...
            "datamodule.batch_size=2",  # Smaller batch size for faster overfit
            # Define the task as a classification of all (1 and 2) vs. 6=building
            "++datamodule.dataset_description.classification_preprocessing_dict={2:1}",
        ]
    )
    train(cfg)

    # Assert that there was a significative improvement i.e. the model learns.
    metrics = _get_metrics_df_from_tmpdir(tmpdir)
    iou = metrics["train/iou_CLASS_building"].dropna()
    improvement = iou.iloc[-1] - iou.iloc[0]
    assert improvement >= 0.55


def test_RandLaNet_test_right_after_training(
    isolated_toy_dataset_tmpdir, one_epoch_trained_RandLaNet_checkpoint, tmpdir
):
    """Run test using the model that was just trained for one epoch.

    Args:
        isolated_toy_dataset_tmpdir (fixture -> str): directory to toy dataset
        one_epoch_trained_RandLaNet_checkpoint (fixture -> str): path to checkpoint of
        a RandLa-Net model that was trained for once epoch at start of test session.
        tmpdir (fixture -> str): temporary directory.

    """

    # Run testing on toy testset with trainer.test(...)
    # function's name is train, but under the hood and thanks to configuration,
    # trainer.test(...) is called.
    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_toy_dataset_tmpdir, tmpdir
    )
    cfg_test_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=test",  # sets task.task_name to "test"
            f"model.ckpt_path={one_epoch_trained_RandLaNet_checkpoint}",
        ]
        + tmp_paths_overrides
    )
    train(cfg_test_using_trained_model)


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
            f"predict.src_las={LAS_SUBSET_FOR_TOY_DATASET}",
            f"predict.output_dir={tmpdir}",
            "predict.probas_to_save=[building,unclassified]",
        ]
        + tmp_paths_overrides
    )
    path_to_output_las = predict(cfg_predict_using_trained_model)

    # Check that predict function generates a predicted LAS
    assert osp.isfile(path_to_output_las)

    # Check the format of the predicted las in terms of extra dimensions
    DIMS_ALWAYS_THERE = ["PredictedClassification", "entropy"]
    DIMS_CHOSEN_IN_CONFIG = ["building", "unclassified"]
    check_las_contains_dims(
        path_to_output_las,
        dims_to_check=DIMS_ALWAYS_THERE + DIMS_CHOSEN_IN_CONFIG,
    )
    DIMS_NOT_THERE = ["ground"]
    check_las_does_not_contains_dims(path_to_output_las, dims_to_check=DIMS_NOT_THERE)

    # check that predict does not change other dimensions
    check_las_invariance(LAS_SUBSET_FOR_TOY_DATASET, path_to_output_las)


def test_run_test_with_trained_model_on_toy_dataset_on_cpu(
    isolated_toy_dataset_tmpdir, tmpdir
):
    _run_test_with_trained_model_on_toy_dataset(
        isolated_toy_dataset_tmpdir, tmpdir, "null"
    )


@RunIf(min_gpus=1)
def test_run_test_with_trained_model_on_toy_dataset_on_gpu(
    isolated_toy_dataset_tmpdir, tmpdir
):
    _run_test_with_trained_model_on_toy_dataset(
        isolated_toy_dataset_tmpdir, tmpdir, "[0]"
    )


def _run_test_with_trained_model_on_toy_dataset(
    isolated_toy_dataset_tmpdir, tmpdir, trainer_gpus
):
    """Test performance (IoU) from a trained model on the toy LAS.

    Check that IoU for building, ground, and unclassified points is above some thresholds.

    Args:
        isolated_toy_dataset_tmpdir (fixture -> str): directory to toy dataset
        tmpdir (fixture -> str): temporary directory.

    """
    if not osp.isfile(TRAINED_MODEL_PATH):
        pytest.xfail(reason=f"No access to {TRAINED_MODEL_PATH} in this environment.")

    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_toy_dataset_tmpdir, tmpdir
    )
    # Use an experiment designed for testing on test set
    cfg_test_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=test",
            f"model.ckpt_path={TRAINED_MODEL_PATH}",
            f"trainer.gpus={trainer_gpus}",
        ]
        + tmp_paths_overrides
    )
    train(cfg_test_using_trained_model)
    metrics = _get_metrics_df_from_tmpdir(tmpdir)
    assert metrics["test/iou_CLASS_unclassified"][0] >= 0.4
    assert metrics["test/iou_CLASS_ground"][0] >= 0.65
    assert metrics["test/iou_CLASS_building"][0] >= 0.60


@pytest.mark.slow()
def test_run_test_with_trained_model_on_large_las(
    isolated_test_subdir_for_large_las, tmpdir
):
    """Test performance (IoU) from a trained model on a large LAS.

    Check that IoU for building, ground, and unclassified points is above some thresholds.

    TODO: remove xfail clause when model memory-consumption allow it.

    Args:
        isolated_test_subdir_for_large_las (fixture -> str): directory to dataset derived from large las
        tmpdir (fixture -> str): temporary directory.

    """
    # pytest.xfail(
    #     reason="Model is currently too memory intensive to test on a full LAS in self-hosted action-runner."
    # )

    if not osp.isfile(TRAINED_MODEL_PATH):
        pytest.xfail(reason=f"No access to {TRAINED_MODEL_PATH} in this environment.")

    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_test_subdir_for_large_las, tmpdir
    )
    # Use an experiment designed for testing on test set
    cfg_test_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=test",
            f"model.ckpt_path={TRAINED_MODEL_PATH}",
        ]
        + tmp_paths_overrides
    )
    train(cfg_test_using_trained_model)
    metrics = _get_metrics_df_from_tmpdir(tmpdir)

    # Reference value of IoU of subsampled cloud point (current method)
    assert metrics["test/iou_CLASS_building"][0] >= 0.74
    assert metrics["test/iou_CLASS_ground"][0] >= 0.77
    assert metrics["test/iou_CLASS_unclassified"][0] >= 0.49

    # Reference value with full interpolation would be
    # assert metrics["test/iou_CLASS_unclassified"][0] >= 0.55
    # assert metrics["test/iou_CLASS_ground"][0] >= 0.80
    # assert metrics["test/iou_CLASS_building"][0] >= 0.80


def test_predict_with_trained_model_on_toy_dataset(tmpdir):
    """Simple check that prediction does not fail when running from a trained model.

    Args:
        tmpdir (fixture -> str): temporary directory.

    """
    if not osp.isfile(TRAINED_MODEL_PATH):
        pytest.xfail(reason=f"No access to {TRAINED_MODEL_PATH} in this environment.")

    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        "placeholder_because_no_need_for_a_dataset_here", tmpdir
    )
    cfg_predict_using_trained_model = make_default_hydra_cfg(
        overrides=[
            "experiment=predict",
            f"predict.ckpt_path={TRAINED_MODEL_PATH}",
            f"predict.src_las={LAS_SUBSET_FOR_TOY_DATASET}",
            f"predict.output_dir={tmpdir}",
            "predict.probas_to_save=[building,unclassified]",
        ]
        + tmp_paths_overrides
    )
    output_las_path = predict(cfg_predict_using_trained_model)
    assert osp.isfile(output_las_path)


# @pytest.mark.slow()
# def test_predict_with_trained_model_on_large_las(make_default_hydra_cfg):
#     if not osp.isfile(TRAINED_MODEL_PATH):
#         pytest.xfail(reason=f"No access to {TRAINED_MODEL_PATH} in this environment.")
#     if not osp.isfile(LARGE_LAS_PATH):
#         pytest.xfail(reason=f"No access to {LARGE_LAS_PATH} in this environment.")

#     with TemporaryDirectory() as tmpdir:
#         hydra_overrides = make_list_of_hydra_overrides_for_logger_and_paths(
#             "placeholder_because_no_need_for_a_dataset_here", tmpdir
#         )
#         cfg_predict_using_trained_model = make_default_hydra_cfg(
#             overrides=[
#                 f"predict.ckpt_path={TRAINED_MODEL_PATH}",
#                 f"predict.src_las={LARGE_LAS_PATH}",
#                 f"predict.output_dir={tmpdir}",
#                 "predict.probas_to_save=[building,unclassified]",
#             ]
#             + hydra_overrides
#         )
#         output_las_path = predict(cfg_predict_using_trained_model)
#         assert osp.isfile(output_las_path)

#         # TODO: compare result to Classification
#         array = pdal_read_las_array(output_las_path)
#         target = array["Classification"]
#         preds = array["PredictedClassification"]
#         iou = torchmetrics.JaccardIndex(num_classes=3, absent_score=1.0)
#         assert iou(torch.Tensor(preds), torch.Tensor(target)) >= 0.8
#         assert np.mean(target == preds) >= 0.97


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


def pdal_read_las_array(las_path: str):
    """Read LAS as a named array.

    Args:
        las_path (str): input LAS path

    Returns:
        np.ndarray: named array with all LAS dimensions, including extra ones, with dict-like access.

    """
    p1 = pdal.Pipeline() | pdal.Reader.las(filename=las_path)
    p1.execute()
    return p1.arrays[0]


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
    isolated_toy_dataset_tmpdir: str, tmpdir: str
):
    """Get list of overrides for hydra, the ones that are always needed when calling train/test.

    Args:
        isolated_test_subdir_for_large_las (str): path to directory to dataset derived from large las
        tmpdir (str): path to temporary directory.

    """

    return [
        f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
        "logger=csv",  # disables comet logging
        f"logger.csv.save_dir={tmpdir}",
        f"callbacks.model_checkpoint.dirpath={tmpdir}",
    ]


def _get_metrics_df_from_tmpdir(tmpdir: str) -> pd.DataFrame:
    """Get dataframe of metrics logged by csv logger.

    Not sure if version_0 is added by pytest or by lightning.

    """
    return pd.read_csv(osp.join(tmpdir, "csv", "version_0", "metrics.csv"))


@pytest.fixture(scope="session")
def one_epoch_trained_RandLaNet_checkpoint(isolated_toy_dataset_tmpdir, tmpdir_factory):
    """Train a RandLaNet model for one epoch, in order to run it in different other tests.

    Args:
        isolated_toy_dataset_tmpdir (str): path to isolated toy dataset as created by fixture.
        tmpdir_factory (fixture): factory to create a session level tempdir.

    Returns:
        str: path to trained model checkpoint, which persists for the whole pytest session.

    """
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")

    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_toy_dataset_tmpdir, tmpdir
    )
    cfg_one_epoch = make_default_hydra_cfg(
        overrides=[
            "experiment=RandLaNetDebug",
            "datamodule.batch_size=2",
            "trainer.min_epochs=1",
            "trainer.max_epochs=1",
        ]
        + tmp_paths_overrides
    )
    trainer = train(cfg_one_epoch)
    return trainer.checkpoint_callback.best_model_path


@RunIf(min_gpus=1)
def test_one_epoch_RandLaNet_training_using_gpu(
    isolated_toy_dataset_tmpdir, tmpdir_factory
):
    """Train a RandLaNet model for one epoch using GPU. XFail is no GPU available.

    Args:
        isolated_toy_dataset_tmpdir (str): path to isolated toy dataset as created by fixture.
        tmpdir_factory (fixture): factory to create a session-level tempdir.

    """
    tmpdir = tmpdir_factory.mktemp("training_logs_dir")

    tmp_paths_overrides = _make_list_of_necesary_hydra_overrides_with_tmp_paths(
        isolated_toy_dataset_tmpdir, tmpdir
    )

    # We will always use the first GPU id for tests, because it always exists if there are some GPUs.
    # Attention to concurrency with other processes using the GPU when running tests.
    gpu_id = 0
    cfg_one_epoch = make_default_hydra_cfg(
        overrides=[
            "experiment=RandLaNetDebug",
            "datamodule.batch_size=2",
            "trainer.min_epochs=1",
            "trainer.max_epochs=1",
            f"trainer.gpus=[{gpu_id}]",
        ]
        + tmp_paths_overrides
    )
    train(cfg_one_epoch)
