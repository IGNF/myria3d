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
    MetricsNames,
    ShapeFileCols,
    apply_constraint_and_sort,
    derive_raw_shape_level_indicators,
    make_decisions,
    evaluate_decisions,
)

log = utils.get_logger(__name__)

# Use HPO if more than two params : https://github.com/ashleve/lightning-hydra-template#hyperparameter-search
def tune(config: DictConfig) -> Optional[float]:
    """Contains tuning pipeline which takes a building validation shapefile and change decision threshold.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    inspection_shp_path = config.validation_module.comparison_shapefile_path
    assert inspection_shp_path.endswith(".shp")
    pts_level_info_csv_path = inspection_shp_path.replace(".shp", ".csv")
    log.info(f"Evaluation of validation tools from : {inspection_shp_path}")
    df = pd.read_csv(
        pts_level_info_csv_path,
        converters={"BuildingsProba": eval, "TruePositive": eval},
    )
    df_hparams_opti = pd.DataFrame()
    df = derive_raw_shape_level_indicators(
        df,
        min_confidence_confirmation=config.validation_module.min_confidence_confirmation,
        min_confidence_refutation=config.validation_module.min_confidence_refutation,
    )
    df = make_decisions(
        gdf=df,
        min_frac_confirmation=config.validation_module.min_frac_confirmation,
        min_frac_refutation=config.validation_module.min_frac_refutation,
    )
    params = {
        "min_frac_confirmation": config.validation_module.min_frac_confirmation,
        "min_frac_refutation": config.validation_module.min_frac_refutation,
        "min_confidence_confirmation": config.validation_module.min_confidence_confirmation,
        "min_confidence_refutation": config.validation_module.min_confidence_refutation,
    }
    metrics_dict = evaluate_decisions(df)
    metrics_dict.update(params)
    df_hparams_opti = df_hparams_opti.append(metrics_dict, ignore_index=True)
    df_hparams_opti = apply_constraint_and_sort(
        df_hparams_opti,
        minimal_confirmation_accuracy_threshold=config.validation_module.minimal_confirmation_accuracy_threshold,
        minimal_refutation_accuracy_threshold=config.validation_module.minimal_refutation_accuracy_threshold,
    )
    if len(df_hparams_opti) == 0:
        return 0
    PA = df_hparams_opti[MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value].values[0]
    if PA is None:
        print("error")
    # TODO: two params opti later.
    # CA = df_hparams_opti[MetricsNames.CONFIRMATION_ACCURACY.value].values[0]
    # RA = df_hparams_opti[MetricsNames.REFUTATION_ACCURACY.value].values[0]
    # AA = CA*RA
    return PA
