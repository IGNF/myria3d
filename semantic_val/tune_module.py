import os
from typing import List, Optional
import geopandas
import numpy as np

from omegaconf import DictConfig
import pandas as pd
from pytorch_lightning import seed_everything
import glob
import os.path as osp
from tqdm import tqdm
from semantic_val.utils import utils

from semantic_val.validation.validation_utils import (
    ShapeFileCols,
    apply_constraint_and_sort,
    make_decisions,
    evaluate_decisions,
)

log = utils.get_logger(__name__)

# Use HPO if more than two params : https://github.com/ashleve/lightning-hydra-template#hyperparameter-search
def tune_module(config: DictConfig) -> Optional[float]:
    """Contains tuning pipeline which takes a building validation shapefile and change decision threshold.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    module_shp_filepath = config.validation_module.comparison_shapefile_path
    log.info(f"Evaluation of validation tools from : {module_shp_filepath}")
    gdf = geopandas.read_file(module_shp_filepath)

    df_hparams_opti = pd.DataFrame()
    for validation_threshold in np.linspace(start=0.0, stop=1.0, num=20):
        for refutation_threshold in np.linspace(start=0.0, stop=1.0, num=20):
            gdf = make_decisions(
                gdf=gdf,
                validation_threshold=validation_threshold,
                refutation_threshold=refutation_threshold,
            )
            metrics_dict = evaluate_decisions(gdf)
            metrics_dict.update({"validation_threshold": validation_threshold})
            metrics_dict.update({"refutation_threshold": refutation_threshold})
            df_hparams_opti = df_hparams_opti.append(metrics_dict, ignore_index=True)

    df_hparams_opti = apply_constraint_and_sort(
        df_hparams_opti,
        minimal_confirmation_accuracy_threshold=config.validation_module.minimal_confirmation_accuracy_threshold,
        minimal_refutation_accuracy_threshold=config.validation_module.minimal_refutation_accuracy_threshold,
    )
    log.info("Top three sets of hparams meeting constraints: ")
    log.info("\n" + str(df_hparams_opti.iloc[:3]))
    hparams_opti_filepath = osp.join(
        osp.dirname(config.validation_module.comparison_shapefile_path),
        config.validation_module.hparams_opti_output_csv_path,
    )
    log.info(f"Saving hparams optimization results to {hparams_opti_filepath}")
    df_hparams_opti.to_csv(hparams_opti_filepath, index=False)
