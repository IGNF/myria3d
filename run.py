try:
    # It is safer to import comet before all other imports.
    import comet_ml  # noqa
except ImportError:
    print("Warning: package comet_ml not found. This may break things if you use a comet callback.")

from enum import Enum

import os
import sys
from glob import glob
import dotenv
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from myria3d.utils import utils
from myria3d.pctl.dataset.hdf5 import create_hdf5
from myria3d.pctl.dataset.utils import get_las_paths_by_split_dict

TASK_NAME_DETECTION_STRING = "task.task_name="
DEFAULT_DIRECTORY = "trained_model_assets/"
DEFAULT_CONFIG_FILE = "proto151_V2.0_epoch_100_Myria3DV3.1.0_predict_config_V3.2.0.yaml"
DEFAULT_CHECKPOINT = "proto151_V2.0_epoch_100_Myria3DV3.1.0.ckpt"
DEFAULT_ENV = "placeholder.env"


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

    # hydra changes current directory, so we make sure the checkpoint has an absolute path
    if not os.path.isabs(config.predict.ckpt_path):
        config.predict.ckpt_path = os.path.join(os.path.dirname(__file__), config.predict.ckpt_path)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    # Iterate over the files and predict.
    src_las_iterable = glob(config.predict.src_las)
    for config.predict.src_las in tqdm(src_las_iterable):
        predict(config)


@hydra.main(config_path="configs/", config_name="config.yaml")
def launch_hdf5(config: DictConfig):
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
            # load environment variables from `.env` file if it exists
            # recursively searches for `.env` in all folders starting from work dir
            dotenv.load_dotenv(override=True)
            launch_train()

        elif task_name == TASK_NAMES.PREDICT.value:
            dotenv.load_dotenv(os.path.join(DEFAULT_DIRECTORY, DEFAULT_ENV))
            launch_predict()

        elif task_name == TASK_NAMES.HDF5.value:
            launch_hdf5()

        else:
            log.warning("Task unknown")

    except NameError as e:
        log.error('a task name must be defined, with the argument "task.task_name=..."')
        raise e
