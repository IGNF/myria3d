try:
    # It is safer to import comet before all other imports.
    import comet_ml  # noqa
except ImportError:
    print("Warning: package comet_ml not found. This may break things if you use a comet callback.")

from enum import Enum

import os
import sys
import dotenv
import hydra
from omegaconf import DictConfig

from myria3d.pctl.dataset.hdf5 import create_hdf5
from myria3d.pctl.dataset.utils import get_las_paths_by_split_dict

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

DEFAULT_DIRECTORY = "default_files_for_predict"
DEFAULT_CONFIG_FILE = "default_config.yaml"


class TASK_NAMES(Enum):
    FIT = "fit"
    TEST = "test"
    FINETUNE = "finetune"
    PREDICT = "predict"
    HDF5 = "create_hdf5"


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    """Entrypoint to training-related logic and inference code.

    Run `python run.py -h` to print configuration and see parameters.

    Hydra configs can be overriden in CLI with `--config-path` and `--config-name` arguments.

    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from myria3d.predict import predict
    from myria3d.train import train
    from myria3d.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)
    task_name = config.task.get("task_name")

    if task_name in [TASK_NAMES.FIT.value, TASK_NAMES.TEST.value, TASK_NAMES.FINETUNE.value]:
        """Training, evaluation, testing, or finetuning of a neural network."""
        # Pretty print config using Rich library
        if config.get("print_config"):
            utils.print_config(config, resolve=False)
        return train(config)

    elif task_name == TASK_NAMES.PREDICT.value:
        """Infer probabilities and automate semantic segmentation decisions on unseen data."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path=DEFAULT_DIRECTORY)
        overrides = sys.argv[1:]  # we will use the default config file but we have to transfert the user's overrides to it
        config = hydra.compose(config_name=DEFAULT_CONFIG_FILE, overrides=overrides)
        config.predict.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), config.predict.ckpt_path)
        # Pretty print config using Rich library
        if config.get("print_config"):
            utils.print_config(config, resolve=False)
        return predict(config)

    elif task_name == TASK_NAMES.HDF5.value:
        """Build an HDF5 file from a directory with las files."""
        # Pretty print config using Rich library
        if config.get("print_config"):
            utils.print_config(config, resolve=False)

        las_paths_by_split_dict = get_las_paths_by_split_dict(config.datamodule.get("data_dir"), config.datamodule.get("split_csv_path"))
        create_hdf5(
            las_paths_by_split_dict=las_paths_by_split_dict,
            hdf5_file_path=config.datamodule.get("hdf5_file_path"),
            tile_width=config.datamodule.get("tile_width"),
            subtile_width=config.datamodule.get("subtile_width"),
            subtile_shape=config.datamodule.get("subtile_shape"),
            pre_filter=hydra.utils.instantiate(config.datamodule.get("pre_filter")),
            subtile_overlap_train=config.datamodule.get("subtile_overlap_train"),
            points_pre_transform=hydra.utils.instantiate(config.datamodule.get("points_pre_transform"))
        )


if __name__ == "__main__":
    main()
