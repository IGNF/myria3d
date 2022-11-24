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

from myria3d.utils import utils
from myria3d.pctl.dataset.hdf5 import create_hdf5
from myria3d.pctl.dataset.utils import get_las_paths_by_split_dict

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
# dotenv.load_dotenv(override=True)

TASK_NAME_DETECTION_STRING = "task.task_name="
DEFAULT_DIRECTORY = "default_files_for_predict"
DEFAULT_CONFIG_FILE = "default_config.yaml"
DEFAULT_CHECKPOINT = "default_checkpoint.ckpt"
DEFAULT_ENV = "default.env"


class TASK_NAMES(Enum):
    FIT = "fit"
    TEST = "test"
    FINETUNE = "finetune"
    PREDICT = "predict"
    HDF5 = "create_hdf5"


log = utils.get_logger(__name__)


@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_train(config: DictConfig):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
    """Training, evaluation, testing, or finetuning of a neural network."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from myria3d.train import train
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)
    return train(config)


@hydra.main(config_path=DEFAULT_DIRECTORY, config_name=DEFAULT_CONFIG_FILE)
def launch_predict(config: DictConfig):
    """Infer probabilities and automate semantic segmentation decisions on unseen data."""
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from myria3d.predict import predict

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=DEFAULT_DIRECTORY)
    overrides = sys.argv[2:]  # we will use the default config file but we have to transfert the user's overrides to it
    new_chekpoint_path = os.path.join(hydra.utils.get_original_cwd(), DEFAULT_DIRECTORY, DEFAULT_CHECKPOINT)
    overrides.append(f"predict.ckpt_path={new_chekpoint_path}")
    config = hydra.compose(config_name=DEFAULT_CONFIG_FILE, overrides=overrides)
    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)
    return predict(config)


@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_hdf5(config: DictConfig):  # pragma: no cover  (it's just an initialyzer of a class/method tested elsewhere)
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
    for arg in sys.argv:
        if TASK_NAME_DETECTION_STRING in arg:
            _, task_name = arg.split("=")
            break

    try:
        log.info(f"task selected: {task_name}")

        if task_name in [TASK_NAMES.FIT.value, TASK_NAMES.TEST.value, TASK_NAMES.FINETUNE.value]:
            dotenv.load_dotenv(override=True)
            launch_train()

        elif task_name == TASK_NAMES.PREDICT.value:
            dotenv.load_dotenv(os.path.join(DEFAULT_DIRECTORY, DEFAULT_ENV))
            launch_predict()

        elif task_name == TASK_NAMES.HDF5.value:
            launch_hdf5()

    except NameError as e:
        log.error(
            'a task name must be defined, with the argument "task.task_name=..."'
        )
        raise e
