from typing import Tuple
from omegaconf import DictConfig
import pandas as pd
from pytorch_lightning import seed_everything
from semantic_val.utils import utils
from semantic_val.validation.validation_utils import (
    MetricsNames,
    change_filepath_suffix,
    derive_shape_indicators,
    make_decisions,
    evaluate_decisions,
)

log = utils.get_logger(__name__)


def tune(config: DictConfig) -> Tuple[float]:
    """Take inspection shapes and make decision based on configurable thresholds.
    Called in a parameters sweep as part of hydra + optuna.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    pts_level_info_csv_path = change_filepath_suffix(
        config.validation_module.comparison_shapefile_path
    )
    log.debug(f"Evaluation of inspection - {pts_level_info_csv_path}")

    df = pd.read_csv(
        pts_level_info_csv_path,
        converters={"BuildingsProba": eval, "TruePositive": eval},
    )
    df = derive_shape_indicators(
        df,
        min_confidence_confirmation=config.validation_module.min_confidence_confirmation,
        min_confidence_refutation=config.validation_module.min_confidence_refutation,
    )
    df = make_decisions(
        gdf=df,
        min_frac_confirmation=config.validation_module.min_frac_confirmation,
        min_frac_refutation=config.validation_module.min_frac_refutation,
    )
    metrics_dict = evaluate_decisions(df)

    PA = metrics_dict[MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value]
    CA = metrics_dict[MetricsNames.CONFIRMATION_ACCURACY.value]
    RA = metrics_dict[MetricsNames.REFUTATION_ACCURACY.value]
    log.info(f"--------> PA={PA:.3} | RA={RA:.3}  CA={CA:.3}")
    return PA, RA, CA
