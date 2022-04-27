import os.path as osp
from tempfile import TemporaryDirectory
import pandas as pd
import pytest
from lidar_multiclass.train import train
from tests.conftest import run_hydra_decorated_command

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
        cfg = make_default_hydra_cfg(
            overrides=[
                "experiment=RandLaNetDebug",  # Use an experiment designe for overfitting a batch
                # "model=point_net_model",  # but use PointNet model instead of RandLaNet
                "logger=csv",  # disables comet logging
                "datamodule.batch_size=2",  # Smaller batch size for faster overfit
                "datamodule.num_workers=2",  # More workers to avoid bottleneck
                f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
                f"logger.csv.save_dir={tmpdir}",
            ]
        )
        train(cfg)
        # Not sure if version_0 is added by pytest or by lightning, but it is needed.
        metrics = pd.read_csv(osp.join(tmpdir, "csv", "version_0", "metrics.csv"))
        iou = metrics["train/iou_CLASS_building"].dropna()
        improvement = iou.iloc[-1] - iou.iloc[0]
        assert improvement >= 0.75
