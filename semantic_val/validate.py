import os
from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import glob
import os.path as osp
from tqdm import tqdm
from semantic_val.validation.validation_utils import get_inspection_shapefile
from semantic_val.utils import utils

log = utils.get_logger(__name__)


def validate(config: DictConfig) -> Optional[float]:
    """
    Contains validation pipeline, which takes predicted las and generate a single,
    large shapefile with vectorized candidate building and the trained model
    decision (confirm/refute).

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    log.info(f"Logging directory: {os.getcwd()}")

    las_filepath = glob.glob(
        osp.join(config.validation_module.predicted_las_dirpath, "*.las")
    )
    output_shp = osp.join(
        os.getcwd(), config.validation_module.operationnal_output_shapefile_name
    )
    for las_filepath in tqdm(las_filepath, desc="Evaluating predicted point cloud"):
        log.info(f"Evaluation of tile {las_filepath}...")
        df_out = get_inspection_shapefile(
            las_filepath,
            confirmation_threshold=config.validation_module.confirmation_threshold,
            refutation_threshold=config.validation_module.refutation_threshold,
        )
        if df_out is not None:
            mode = "w" if not osp.isfile(output_shp) else "a"
            df_out.to_file(output_shp, mode=mode, index=False)

    log.info(f"Output shapefile is in {output_shp}")
