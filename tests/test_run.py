import runpy
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.functional
relative_output_dir = "tmp"

# Default config points to previous model checkpoint 

# Specific Hydra config (directory + name, without the ".yaml" extension) and checkpoint
# to use for the externalized-embeddings prediction test. Adjust these to point at your
# externalized-embeddings assets.
EXTERNAL_EMBEDDINGS_CONFIG_DIR = "trained_model_assets"


# add a method that matches ckpt with config file and returns a list of tuples with 
# (config_file, ckpt_file) for each model in the trained_model_assets directory
def get_model_assets():
    """
        Returns a list of tuples with (config_file, ckpt_file) for each model 
        in the trained_model_assets directory.
    """
    model_assets = []
    for ckpt_file in Path(EXTERNAL_EMBEDDINGS_CONFIG_DIR).glob("*.ckpt"):
        # try to find yaml with ckpt prefix and -inference-Myria3DV*.yaml suffix using glob
        config_file = list(Path(EXTERNAL_EMBEDDINGS_CONFIG_DIR).glob(ckpt_file.stem + "-inference-Myria3DV*.yaml"))
        if config_file:
            # read config file and get subtile_width value from config file
            with open(config_file[0], "r") as f:
                model_assets.append((config_file[0], ckpt_file))
        else:
            raise ValueError(f"No config file found for checkpoint {ckpt_file}")
    return model_assets

def _predict_test_cases():
    """Shared (test_case, specific_args) parametrization for the predict tests.
    """
    return [
        ("default", []),
        (
            "with_overlap",
            [
                "predict.subtile_overlap=25",
                "predict.interpolator.probas_to_save=[building,ground]",
                "task.task_name=predict",
            ],
        ),
    ]


@pytest.mark.parametrize("test_case, specific_args", _predict_test_cases())
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



EXTERNAL_EMBEDDINGS_AVAILABLE = len(get_model_assets()) > 0


def _all_predict_cases():
    if not EXTERNAL_EMBEDDINGS_AVAILABLE:
        return 
    else:
        for config_file, ckpt_file in get_model_assets():
            for test_case, specific_args in _predict_test_cases():
                yield pytest.param(
                    config_file,
                    ckpt_file,
                    test_case,
                    specific_args,
                    id=f"{config_file.stem}-{test_case}",
                )



@pytest.mark.skipif(
    not EXTERNAL_EMBEDDINGS_AVAILABLE,
    reason="Externalized-embeddings model assets are not available (not published to the repo).",
)
@pytest.mark.parametrize(
    "config_file, ckpt_file, test_case, specific_args", list(_all_predict_cases())
)
def test_run_predict_externalized_embeddings_with_tile_size(
    monkeypatch, config_file, ckpt_file, test_case, specific_args
):
    """Test run.py execution when __name__ == "__main__" with mocked command line input."""
    test_args = [
        "run.py",
        # Hydra CLI flags selecting a specific config directory + name. These must come
        # before any key=value overrides.
        f"--config-dir={EXTERNAL_EMBEDDINGS_CONFIG_DIR}",
        f"--config-name={config_file.stem}",
        "predict.src_las=${hydra:runtime.cwd}/tests/data/las_subset_buildings/792000_6272000_subset_buildings.laz",
        "predict.output_dir=${hydra:runtime.cwd}" + f"/{relative_output_dir}/{test_case}",
        f"predict.ckpt_path={ckpt_file}",
        "task.task_name=predict",
    ] + specific_args
    monkeypatch.setattr(sys, "argv", test_args)
    runpy.run_module("run", run_name="__main__", alter_sys=True)
    assert Path(f"{relative_output_dir}/{test_case}/792000_6272000_subset_buildings.laz").is_file()
