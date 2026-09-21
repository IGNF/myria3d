import runpy
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.functional
relative_output_dir = "tmp"

# Specific Hydra config (directory + name, without the ".yaml" extension) and checkpoint
# to use for the externalized-embeddings prediction test. Adjust these to point at your
# externalized-embeddings assets.
EXTERNAL_EMBEDDINGS_CONFIG_DIR = "trained_model_assets"
EXTERNAL_EMBEDDINGS_CONFIG_NAME = "Myria3d_3.9.2_100mx100m_config_point_budget_120k.yaml"
EXTERNAL_EMBEDDINGS_CKPT_PATH = "trained_model_assets/LargeDataset_100K_100MPatch_LargeRandlanet_epoch_072.ckpt"

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

@pytest.mark.parametrize(
    "test_case, specific_args",
    [
        ("default", []),
        (
            "with_overlap",
            [
                "predict.subtile_overlap=25",
                "datamodule.batch_size=4",
                "predict.interpolator.probas_to_save=[building,ground]",
                "task.task_name=predict",
            ],
        ),
    ],
)
def test_run_predict_externalized_embeddings_with_tile_size(monkeypatch, test_case, specific_args):
    """Test run.py execution when __name__ == "__main__" with mocked command line input."""
    test_args = [
        "run.py",
        # Hydra CLI flags selecting a specific config directory + name. These must come
        # before any key=value overrides.
        f"--config-dir={EXTERNAL_EMBEDDINGS_CONFIG_DIR}",
        f"--config-name={EXTERNAL_EMBEDDINGS_CONFIG_NAME}",
        "+model.neural_net_hparams.dims=[48,192,384,768]",
        "predict.src_las=${hydra:runtime.cwd}/tests/data/las_subset_buildings/792000_6272000_subset_buildings.laz",
        "predict.output_dir=${hydra:runtime.cwd}" + f"/{relative_output_dir}/{test_case}",
        f"predict.ckpt_path={EXTERNAL_EMBEDDINGS_CKPT_PATH}",
        "task.task_name=predict",
    ] + specific_args
    monkeypatch.setattr(sys, "argv", test_args)
    runpy.run_module("run", run_name="__main__", alter_sys=True)
    assert Path(f"{relative_output_dir}/{test_case}/792000_6272000_subset_buildings.laz").is_file()
