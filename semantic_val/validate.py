from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from semantic_val.utils import utils

log = utils.get_logger(__name__)


def validate(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Get the LAS files
    print(config.validation_module.predicted_las_dirpath)

    # Get the shapefile (geopandas)

    # Not optimized :
    # Have a dict with shapes and associated points
    # For each LAS file, add all points of LAS to associated shape, if any (maybe filter once first then associate)
    # Average points in shape, flagging shape with results in the process. (assume that there are no shapes at borders of tile)
