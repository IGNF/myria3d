from typing import Tuple
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
from semantic_val.datamodules.processing import ChannelNames

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
    results = [metrics_dict[metric_enum.value] for metric_enum in MetricsNames]
    results_logs = "  |  ".join(
        f"{metric_enum.value}={value:.3}"
        for metric_enum, value in zip(MetricsNames, results)
    )
    log.info(f"--------> {results_logs}")

    # Optimize subset of metrics specified in config.inspection.metrics
    results = [
        metrics_dict[MetricsNames[metric_name].value]
        for metric_name in config.inspection.metrics
    ]
    return results
