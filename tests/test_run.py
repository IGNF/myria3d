import runpy
import sys

import pytest

pytestmark = pytest.mark.functional


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
        "predict.src_las=/inputs/792000_6272000_subset_buildings.laz",
        f"predict.output_dir=/outputs/{test_case}",
        "task.task_name=predict",
    ] + specific_args
    monkeypatch.setattr(sys, "argv", test_args)
    runpy.run_module("run", run_name="__main__", alter_sys=True)
