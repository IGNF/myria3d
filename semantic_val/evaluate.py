import os
from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import glob
import os.path as osp
from tqdm import tqdm
from semantic_val.utils import utils

from semantic_val.validation.validation_utils import ShapeFileCols

log = utils.get_logger(__name__)


def evaluate(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    log.info(
        f"Evaluation of validation tools in : {config.validation_module.comparison_shapefile_path}"
    )
    print(ShapeFileCols)
