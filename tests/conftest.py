import os
import os.path as osp
import shutil
import pytest
import sh
from typing import List
from hydra.experimental import compose, initialize

from lidar_multiclass.data.loading import FrenchLidarDataLogic


@pytest.fixture
def default_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config")


LAS_SUBSET = "tests/data/870000_6618000.subset.50mx100m.las"
SPLIT_CSV = "tests/data/toy_dataset_split.csv"


@pytest.fixture(scope="session")
def isolated_toy_dataset_tmpdir(tmpdir_factory):
    # Create session scope directory
    tmpdir = tmpdir_factory.mktemp("toy_dataset_tmpdir")

    # Copy input file for full test isolation
    td_raw = osp.join(tmpdir, "raw")
    td_prepared = osp.join(tmpdir, "prepared")
    os.makedirs(td_raw)
    os.makedirs(td_prepared)
    # Make a "raw", unporcessed dataset with six files.
    basename = osp.basename(LAS_SUBSET)
    for s in ["train1", "train2", "val1", "val2", "test1", "test2"]:
        copy_path = osp.join(td_raw, basename.replace(".las", f".{s}.las"))
        shutil.copy(LAS_SUBSET, copy_path)

    # Prepare a Deep-Learning-ready dataset, using the split defined in the csv.
    data_prepper = FrenchLidarDataLogic(
        input_data_dir=td_raw,
        prepared_data_dir=td_prepared,
        split_csv=SPLIT_CSV,
        input_tile_width_meters=110,
        subtile_width_meters=50,
    )
    data_prepper.prepare()
    return td_prepared


def run_command(command: List[str]):
    """Default method for executing shell commands with pytest."""
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(reason=msg)


def run_hydra_decorated_command(command: List[str]):
    """Default method for executing hydra decorated shell commands with pytest."""
    hydra_specific_paths = [
        "hydra.run.dir=tests/logs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "hydra.sweep.dir=tests/logs/multiruns/${now:%Y-%m-%d}/${now:%H-%M-%S}",
    ]
    run_command(command + hydra_specific_paths)
