from typing import Tuple
import numpy as np
from omegaconf import DictConfig
import pandas as pd
from pytorch_lightning import seed_everything
from semantic_val.utils import utils
from semantic_val.inspection.utils import (
    MetricsNames,
    change_filepath_suffix,
    derive_shape_ground_truths,
    derive_shape_indicators,
    make_decisions,
    evaluate_decisions,
    TRUE_POSITIVES_COLNAME,
)
from semantic_val.callbacks.predictions_callbacks import ChannelNames

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

    csv_path = change_filepath_suffix(
        config.inspection.comparison_shapefile_path, ".shp", ".csv"
    )
    log.debug(f"Evaluation of inspection using: {csv_path}")

    points_gdf = pd.read_csv(
        csv_path,
        converters={
            ChannelNames.BuildingsProba.value: eval,
            TRUE_POSITIVES_COLNAME: eval,
        },
    )
    points_gdf = derive_shape_ground_truths(points_gdf)
    points_gdf = derive_shape_indicators(
        points_gdf,
        min_confidence_confirmation=config.inspection.min_confidence_confirmation,
        min_confidence_refutation=config.inspection.min_confidence_refutation,
    )
    points_gdf = make_decisions(
        gdf=points_gdf,
        min_frac_confirmation=config.inspection.min_frac_confirmation,
        min_frac_refutation=config.inspection.min_frac_refutation,
    )
    metrics_dict = evaluate_decisions(points_gdf)

    # print all metrics
    results_logs = "  |  ".join(
        f"{metric_enum.value}={metrics_dict[metric_enum.value]:{'' if type(metrics_dict[metric_enum.value]) is int else '.3'}}"
        for metric_enum in MetricsNames
        if metric_enum
        not in [
            MetricsNames.CONFUSION_MATRIX_NORM,
            MetricsNames.CONFUSION_MATRIX_NO_NORM,
        ]
    )
    results_logs = (
        results_logs
        + "\n"
        + str(metrics_dict[MetricsNames.CONFUSION_MATRIX_NO_NORM.value].round(3))
        + "\n"
        + str(metrics_dict[MetricsNames.CONFUSION_MATRIX_NORM.value].round(3))
    )
    log.info(f"--------> {results_logs}")

    # Optimize subset of metrics specified in config.inspection.metrics
    results = [
        metrics_dict[MetricsNames[metric_name].value]
        for metric_name in config.inspection.metrics
    ]
    if any(np.isnan(m) for m in results):
        # This can occur for trivial solutions with unique decision
        return [0 for _ in results]

    return results
