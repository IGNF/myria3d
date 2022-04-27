import pytest
import sh
from typing import List
from hydra import compose, initialize

from lidar_multiclass.data.loading import make_toy_dataset_from_test_file


@pytest.fixture()
def make_default_hydra_cfg():
    def _make_default_hydra_cfg(overrides=[]):
        with initialize(config_path="./../configs/", job_name="config"):
            # there is no hydra:runtime.cwd when using compose, and therefore we have
            # to specify where our working directory is i.e. where the code is.
            workdir_override = ["work_dir=./../"]
            return compose(config_name="config", overrides=workdir_override + overrides)

    return _make_default_hydra_cfg


@pytest.fixture(scope="session")
def isolated_toy_dataset_tmpdir(tmpdir_factory):
    # Create session scope directory
    tmpdir = tmpdir_factory.mktemp("toy_dataset_tmpdir")
    td_prepared = make_toy_dataset_from_test_file(tmpdir)
    return td_prepared


@pytest.fixture(autouse=True)  # Auto-used for every test function
def set_logs_dir_env_variable(monkeypatch):
    """Sets where hydra saves its logs.

    See: https://docs.pytest.org/en/stable/how-to/monkeypatch.html#monkeypatching-environment-variables

    """
    monkeypatch.setenv("LOGS_DIR", "tests/logs/")


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
