import os
import os.path as osp
import shutil
import pytest
from pytorch_lightning import seed_everything
import sh
from typing import List
from hydra import compose, initialize

from myria3d.data.loading import make_toy_dataset_from_test_file

LARGE_LAS_PATH = "tests/data/large/raw_792000_6272000.las"
TRAINED_MODEL_PATH = "tests/data/large/RandLaNet_Buildings_B2V0.5_epoch_033.ckpt"


@pytest.fixture(scope="session")
def isolated_toy_dataset_tmpdir(tmpdir_factory):
    """Creates a toy dataset accessible from within the pytest session."""
    tmpdir = tmpdir_factory.mktemp("toy_dataset_tmpdir")
    tmpdir_prepared = make_toy_dataset_from_test_file(tmpdir)
    return tmpdir_prepared


@pytest.fixture(scope="session")
def isolated_test_subdir_for_large_las(tmpdir_factory):
    """Copies the large las in a `test` subdirectory, for it to be used by datamodule."""

    if not osp.isfile(LARGE_LAS_PATH):
        pytest.xfail(reason=f"No access to {LARGE_LAS_PATH} in this environment.")

    dataset_tmpdir = tmpdir_factory.mktemp("dataset_tmpdir")
    tmpdir_test_subdir = osp.join(dataset_tmpdir, "test")
    os.makedirs(tmpdir_test_subdir)
    copy_path = osp.join(tmpdir_test_subdir, osp.basename(LARGE_LAS_PATH))
    shutil.copy(LARGE_LAS_PATH, copy_path)
    return dataset_tmpdir


def make_default_hydra_cfg(overrides=[]):
    """Compose the repository hydra config, with specified overrides."""
    with initialize(config_path="./../configs/", job_name="config"):
        # there is no hydra:runtime.cwd when using compose, and therefore we have
        # to specify where our working directory is.
        workdir_override = ["work_dir=./../"]
        return compose(config_name="config", overrides=workdir_override + overrides)


@pytest.fixture(autouse=True)  # Auto-used for every test function
def set_logs_dir_env_variable(monkeypatch):
    """Sets where hydra saves its logs, as we cannot rely on a .env file for tests.

    See: https://docs.pytest.org/en/stable/how-to/monkeypatch.html#monkeypatching-environment-variables

    """
    monkeypatch.setenv("LOGS_DIR", "tests/logs/")
    # to ignore it when making prediction
    # However, this seems not found when running via CLI.
    monkeypatch.setenv("PREPARED_DATA_DIR", "placeholder")


@pytest.fixture(autouse=True)  # Auto-used for every test function
def seed_everything_in_tests():
    seed_everything(12345, workers=True)


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
