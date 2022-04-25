import pytest
import os
import os.path as osp
import shutil
import tempfile
from lidar_multiclass.data.loading import FrenchLidarDataLogic
from tests.conftest import run_hydra_decorated_command

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""
LAS_SUBSET = "tests/data/870000_6618000.subset.50mx100m.las"
SPLIT_CSV = "tests/data/toy_dataset_split.csv"


@pytest.mark.slow()
def test_fast_dev_run(monkeypatch):
    """Test running for 1 train, val and test batch."""
    # https://docs.pytest.org/en/stable/how-to/monkeypatch.html#monkeypatching-environment-variables
    # Set required environment variables, which enables cleaner command run without a lot of hydra overrides
    # Here, set where hydra saves its logs.
    monkeypatch.setenv("LOGS_DIR", "tests/logs/")

    # Copy input file for full test isolation
    with tempfile.TemporaryDirectory() as td:
        td_raw = osp.join(td, "raw")
        td_prepared = osp.join(td, "prepared")
        os.makedirs(td_raw)
        os.makedirs(td_prepared)
        # Make a "raw", unporcessed dataset with six files.
        basename = osp.basename(LAS_SUBSET)
        for s in ["train1", "train2", "val1", "val2", "test1", "test2"]:
            copy_path = osp.join(td_raw, basename.replace(".las", f".{s}.las"))
            shutil.copy(LAS_SUBSET, copy_path)

        # Prepare the dataset for training, using the split defined in the csv.
        data_prepper = FrenchLidarDataLogic(
            input_data_dir=td_raw,
            prepared_data_dir=td_prepared,
            split_csv=SPLIT_CSV,
            input_tile_width_meters=110,
            subtile_width_meters=50,
        )
        data_prepper.prepare()

        command = [
            "run.py",
            "logger=csv",  # disables logging
            f"datamodule.prepared_data_dir={td_prepared}",
            "++trainer.fast_dev_run=true",
        ]

        run_hydra_decorated_command(command)
