import runpy
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.functional
relative_output_dir = "tmp"


@pytest.mark.parametrize(
    "test_case, specific_args",
    [
        ("default", []),
        (
            "with_overlap",
            [
                "predict.subtile_overlap=25",
                "datamodule.batch_size=10",
                "predict.interpolator.probas_to_save=[building,ground]",
                "task.task_name=predict",
            ],
        ),
    ],
)
def test_run_predict(monkeypatch, test_case, specific_args):
    """Test run.py execution when __name__ == "__main__" with mocked command line input."""
    test_args = [
        "run.py",
        "predict.src_las=${hydra:runtime.cwd}/tests/data/las_subset_buildings/792000_6272000_subset_buildings.laz",
        "predict.output_dir=${hydra:runtime.cwd}" + f"/{relative_output_dir}/{test_case}",
        "task.task_name=predict",
    ] + specific_args
    monkeypatch.setattr(sys, "argv", test_args)
    runpy.run_module("run", run_name="__main__", alter_sys=True)
    assert Path(f"{relative_output_dir}/{test_case}/792000_6272000_subset_buildings.laz").is_file()
