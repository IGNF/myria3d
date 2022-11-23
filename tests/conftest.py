from typing import List
import os

import pytest
import sh
from hydra import compose, initialize
from pytorch_lightning import seed_everything

from myria3d.pctl.dataset.toy_dataset import make_toy_dataset_from_test_file


@pytest.fixture(scope="session")
def toy_dataset_hdf5_path(tmpdir_factory):
    """Creates a toy dataset accessible from within the pytest session."""
    return make_toy_dataset_from_test_file()


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
        "hydra.run.dir=" + os.getcwd(),
    ]

    run_command(command + hydra_specific_paths)
