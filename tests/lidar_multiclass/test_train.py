import pytest
from tests.conftest import run_hydra_decorated_command

"""
A couple of sanity checks to make sure the model doesn't crash with different running options.
"""


@pytest.mark.slow()
def test_fast_dev_run(monkeypatch, isolated_toy_dataset_tmpdir):
    """Test running for 1 train, val and test batch."""
    # https://docs.pytest.org/en/stable/how-to/monkeypatch.html#monkeypatching-environment-variables
    # Set required environment variables, which enables cleaner command run without a lot of hydra overrides
    # Here, set where hydra saves its logs.
    monkeypatch.setenv("LOGS_DIR", "tests/logs/")
    command = [
        "run.py",
        "logger=csv",  # disables logging
        f"datamodule.prepared_data_dir={isolated_toy_dataset_tmpdir}",
        "++trainer.fast_dev_run=true",
        # datamodule.batch_size=3
    ]

    run_hydra_decorated_command(command)
