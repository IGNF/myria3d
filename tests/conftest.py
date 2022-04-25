import pytest
import sh
from typing import List
from hydra.experimental import compose, initialize


@pytest.fixture
def default_hydra_cfg():
    with initialize(config_path="./../configs/", job_name="config"):
        return compose(config_name="config")


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
