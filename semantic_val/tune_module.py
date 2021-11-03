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
    derive_raw_shape_level_indicators,
    make_decisions,
    evaluate_decisions,
)

log = utils.get_logger(__name__)

# TODO: rename everything as "tune"
# Use HPO if more than two params : https://github.com/ashleve/lightning-hydra-template#hyperparameter-search
def tune_module(config: DictConfig) -> Optional[float]:
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
    num = 21
    df_hparams_opti = pd.DataFrame()
    for min_frac_confirmation in np.linspace(start=0.0, stop=1.0, num=num):
        log.info(f"Min frac confirmation : {min_frac_confirmation}")
        for min_frac_refutation in np.linspace(start=0.0, stop=1.0, num=num):
            for min_confidence_confirmation in np.linspace(
                start=0.0, stop=1.0, num=num
            ):
                for min_confidence_refutation in np.linspace(
                    start=0.0, stop=1.0, num=num
                ):
                    df = derive_raw_shape_level_indicators(
                        df,
                        min_confidence_confirmation=min_confidence_confirmation,
                        min_confidence_refutation=min_confidence_refutation,
                    )
                    df = make_decisions(
                        gdf=df,
                        min_frac_confirmation=min_frac_confirmation,
                        min_frac_refutation=min_frac_refutation,
                    )
                    params = {
                        "min_frac_confirmation": min_frac_confirmation,
                        "min_frac_refutation": min_frac_refutation,
                        "min_confidence_confirmation": min_confidence_confirmation,
                        "min_confidence_refutation": min_confidence_refutation,
                    }
                    metrics_dict = evaluate_decisions(df)
                    metrics_dict.update(params)
                    df_hparams_opti = df_hparams_opti.append(
                        metrics_dict, ignore_index=True
                    )
    # TODO: sort by P_AUTO and then A*A
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
