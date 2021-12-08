from enum import Enum
import json
from typing import List
import json

import pdal

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from semantic_val.datamodules.processing import ChannelNames
from semantic_val.utils import utils

log = utils.get_logger(__name__)

# INITIAL TERRA-SOLID CLASSIFICATION CODES
DEFAULT_CODE = 1
MTS_AUTO_DETECTED_CODE = 6
MTS_TRUE_POSITIVE_CODE_LIST = [19]
MTS_FALSE_POSITIVE_CODE_LIST = [20, 110, 112, 114, 115]
MTS_FALSE_NEGATIVE_CODE_LIST = [21]
LAST_ECHO_CODE = 104

CLUSTER_TOLERANCE = 0.5  # meters
CLUSTER_MIN_POINTS = 10
SHARED_CRS = "EPSG:2154"

# FINAL CLASSIFICATION CODES
class FinalClassificationCode(Enum):
    """Points code after decision for further analysis."""

    IA_REFUTED = 33  # refuted

    IA_REFUTED_AND_DB_OVERLAYED = 34  # unsure
    BOTH_UNSURE = 35  # unsure

    IA_CONFIRMED_ONLY = 36  # confirmed
    DB_OVERLAYED_ONLY = 37  # confirmed
    BOTH_CONFIRMED = 38  # confirmed


class DecisionLabels(Enum):
    """String label used for making confusion matrices during optimization/evaluation"""

    UNSURE = "unsure"  # 0
    NOT_BUILDING = "not-building"  # 1
    BUILDING = "building"  # 2


DECISION_LABELS_LIST = [l.value for l in DecisionLabels]

CODE_TO_LABEL_MAPPER = {
    FinalClassificationCode.IA_REFUTED.value: DecisionLabels.NOT_BUILDING.value,
    FinalClassificationCode.IA_REFUTED_AND_DB_OVERLAYED.value: DecisionLabels.UNSURE.value,
    FinalClassificationCode.BOTH_UNSURE.value: DecisionLabels.UNSURE.value,
    FinalClassificationCode.IA_CONFIRMED_ONLY.value: DecisionLabels.BUILDING.value,
    FinalClassificationCode.DB_OVERLAYED_ONLY.value: DecisionLabels.BUILDING.value,
    FinalClassificationCode.BOTH_CONFIRMED.value: DecisionLabels.BUILDING.value,
}


class MetricsNames(Enum):
    # Shapes info
    MTS_SHP_NUMBER = "NUM_SHAPES"
    MTS_SHP_BUILDINGS = "P_MTS_BUILDINGS"
    MTS_SHP_NO_BUILDINGS = "P_MTS_NO_BUILDINGS"
    MTS_SHP_UNSURE = "P_MTS_UNSURE"

    # Amount of each deicison
    PROPORTION_OF_UNCERTAINTY = "P_UNSURE"
    PROPORTION_OF_CONFIRMATION = "P_CONFIRM"
    PROPORTION_OF_REFUTATION = "P_REFUTE"
    CONFUSION_MATRIX_NORM = "CONFUSION_MATRIX_NORM"
    CONFUSION_MATRIX_NO_NORM = "CONFUSION_MATRIX_NO_NORM"

    # To maximize:
    PROPORTION_OF_AUTOMATED_DECISIONS = "P_AUTO"
    PRECISION = "PRECISION"
    RECALL = "RECALL"
    # Constraints:
    CONFIRMATION_ACCURACY = "A_CONFIRM"
    REFUTATION_ACCURACY = "A_REFUTE"

    # TODO: get rid of Net Gain metrics
    # Metainfo to evaluate absolute gain and what is still to inspect
    NET_GAIN_CONFIRMATION = "NG_CONFIRM"
    NET_GAIN_REFUTATION = "NG_REFUTE"


def prepare_las_for_decision(
    input_filepath: str,
    input_bd_topo_shp: str,
    input_bd_parcellaire_shp: str,
    output_filepath: str,
    candidate_building_points_classification_codes: List[int] = [MTS_AUTO_DETECTED_CODE]
    + MTS_TRUE_POSITIVE_CODE_LIST
    + MTS_FALSE_POSITIVE_CODE_LIST,
):
    """Will:
    - Cluster candidates points, thus creating a ClusterId channel (default cluster: 0).
    - Identify points overlayed by a BDTopo shape, thus creating a BDTopoOverlay channel (no overlap: 0).
    """
    candidates_where = (
        "("
        + " || ".join(
            f"Classification == {int(candidat_code)}"
            for candidat_code in candidate_building_points_classification_codes
        )
        + ")"
    )
    _reader = [
        {
            "type": "readers.las",
            "filename": input_filepath,
            "override_srs": SHARED_CRS,
            "nosrs": True,
        }
    ]
    _cluster = [
        {
            "type": "filters.cluster",
            "min_points": CLUSTER_MIN_POINTS,
            "tolerance": CLUSTER_TOLERANCE,  # meters
            "where": candidates_where,
        }
    ]
    _topo_overlay = [
        {
            "type": "filters.ferry",
            "dimensions": f"=>{ChannelNames.BDTopoOverlay.value}",
        },
        {
            "column": "presence",
            "datasource": input_bd_topo_shp,
            "dimension": f"{ChannelNames.BDTopoOverlay.value}",
            "type": "filters.overlay",
        },
    ]
    _parcellaire_overlay = [
        {
            "type": "filters.ferry",
            "dimensions": f"=>{ChannelNames.BDParcellaireOverlay.value}",
        },
        {
            "column": "presence",
            "datasource": input_bd_parcellaire_shp,
            "dimension": f"{ChannelNames.BDParcellaireOverlay.value}",
            "type": "filters.overlay",
        },
    ]
    _writer = [
        {
            "type": "writers.las",
            "filename": output_filepath,
            "forward": "all",  # keep all dimensions based on input format
            "extra_dims": "all",  # keep all extra dims as well
        }
    ]
    pipeline = {
        "pipeline": _reader + _cluster + _topo_overlay + _parcellaire_overlay + _writer
    }
    pipeline = json.dumps(pipeline)
    pipeline = pdal.Pipeline(pipeline)
    pipeline.execute()
    structured_array = pipeline.arrays[0]
    return structured_array


def reset_classification(classification: np.array):
    """
    Set the classification to pre-correction codes. This is not needed for production.
    FP+TP -> set to auto-detected code
    FN -> set to background code
    LAST_ECHO -> set to background code
    """
    candidate_building_points_mask = np.isin(
        classification, MTS_TRUE_POSITIVE_CODE_LIST + MTS_FALSE_POSITIVE_CODE_LIST
    )
    classification[candidate_building_points_mask] = MTS_AUTO_DETECTED_CODE
    forgotten_buillding_points_mask = np.isin(
        classification, MTS_FALSE_NEGATIVE_CODE_LIST
    )
    classification[forgotten_buillding_points_mask] = DEFAULT_CODE
    last_echo_index = classification == LAST_ECHO_CODE
    classification[last_echo_index] = DEFAULT_CODE
    return classification


def make_group_decision(
    probas,
    topo_overlay_bools,
    parcellaire_overlay_bools,
    min_confidence_confirmation: float = 0.6,
    min_frac_confirmation: float = 0.5,
    min_confidence_refutation: float = 0.6,
    min_frac_refutation: float = 0.8,
    min_topo_overlay_confirmation: float = 0.95,
    min_parcellaire_overlay_confirmation: float = 0.95,
):
    """
    Confirm or refute candidate building shape based on fraction of confirmed/refuted points and
    on fraction of points overlayed by a building shape in a database.
    """
    ia_confirmed = (
        np.mean(probas >= min_confidence_confirmation) >= min_frac_confirmation
    )
    ia_refuted = (
        np.mean((1 - probas) >= min_confidence_refutation) >= min_frac_refutation
    )
    topo_overlayed = np.mean(topo_overlay_bools) >= min_topo_overlay_confirmation
    parcellaire_overlayed = (
        np.mean(parcellaire_overlay_bools) >= min_parcellaire_overlay_confirmation
    )
    bd_overlayed = topo_overlayed or parcellaire_overlayed

    if ia_refuted:
        if bd_overlayed:
            return FinalClassificationCode.IA_REFUTED_AND_DB_OVERLAYED.value
        return FinalClassificationCode.IA_REFUTED.value
    if ia_confirmed:
        if bd_overlayed:
            return FinalClassificationCode.BOTH_CONFIRMED.value
        return FinalClassificationCode.IA_CONFIRMED_ONLY.value
    if bd_overlayed:
        return FinalClassificationCode.DB_OVERLAYED_ONLY.value
    return FinalClassificationCode.BOTH_UNSURE.value


def split_idx_by_dim(dim_array):
    """Returns a sequence of arrays of indices of elements sharing the same value in dim_array"""
    idx = np.argsort(dim_array)
    sorted_dim_array = dim_array[idx]
    group_idx = np.array_split(idx, np.where(np.diff(sorted_dim_array) != 0)[0] + 1)
    return group_idx


def update_las_with_decisions(
    las, params, mts_auto_detected_code: int = MTS_AUTO_DETECTED_CODE
):
    """
    Update point cloud classification channel.
    Params is a dict-like object with optimized decision thresholds.
    """

    # 1) Set to default all candidats points
    candidate_building_points_mask = (
        las[ChannelNames.Classification.value] == mts_auto_detected_code
    )
    las[ChannelNames.Classification.value][
        candidate_building_points_mask
    ] = DEFAULT_CODE

    # 2) Decide at the group-level
    split_idx = split_idx_by_dim(las[ChannelNames.ClusterID.value])
    split_idx = split_idx[1:]  # remove unclustered group with ClusterID = 0
    for pts_idx in tqdm(split_idx, desc="Updating LAS."):
        pts = las.points[pts_idx]
        decision_code = make_group_decision(
            pts[ChannelNames.BuildingsProba.value],
            pts[ChannelNames.BDTopoOverlay.value],
            pts[ChannelNames.BDParcellaireOverlay.value],
            **params,
        )
        las[ChannelNames.Classification.value][pts_idx] = decision_code
    return las


def evaluate_decisions(mts_gt, ia_decision):
    """
    Get dict of metrics to evaluate how good module decisions were in reference to ground truths.
    Targets: U=Unsure, N=No (not a building), Y=Yes (building)
    PRedictions : U=Unsure, C=Confirmation, R=Refutation
    Confusion Matrix :
            predictions
            [Uu Ur Uc]
    target  [Nu Nr Nc]
            [Yu Yr Yc]

    Maximization criteria:
      Proportion of each decision among total of candidate groups.
      We want to maximize it.
    Accuracies:
      Confirmation/Refutation Accuracy.
      Accurate decision if either "unsure" or the same as the label.
    Quality
      Precision and Recall, assuming perfect posterior decision for unsure predictions.
      Only candidate shapes with known ground truths are considered (ambiguous labels are ignored).
      Precision :
      Recall : (Yu + Yc) / (Yu + Yn + Yc)
    """
    metrics_dict = dict()

    # VECTORS INFOS
    num_shapes = len(ia_decision)
    metrics_dict.update({MetricsNames.MTS_SHP_NUMBER.value: num_shapes})

    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize=None
    )
    metrics_dict.update({MetricsNames.CONFUSION_MATRIX_NO_NORM.value: cm.copy()})

    # CRITERIA
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="all"
    )
    P_MTS_U, P_MTS_N, P_MTS_C = cm.sum(axis=1)
    metrics_dict.update(
        {
            MetricsNames.MTS_SHP_UNSURE.value: P_MTS_U,
            MetricsNames.MTS_SHP_NO_BUILDINGS.value: P_MTS_N,
            MetricsNames.MTS_SHP_BUILDINGS.value: P_MTS_C,
        }
    )
    P_IA_u, P_IA_r, P_IA_c = cm.sum(axis=0)
    PAD = P_IA_c + P_IA_r
    metrics_dict.update(
        {
            MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value: PAD,
            MetricsNames.PROPORTION_OF_UNCERTAINTY.value: P_IA_u,
            MetricsNames.PROPORTION_OF_REFUTATION.value: P_IA_r,
            MetricsNames.PROPORTION_OF_CONFIRMATION.value: P_IA_c,
        }
    )

    # CONSTRAINTS
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="pred"
    )
    RA = cm[1, 1]
    CA = cm[2, 2]
    metrics_dict.update(
        {
            MetricsNames.REFUTATION_ACCURACY.value: RA,
            MetricsNames.CONFIRMATION_ACCURACY.value: CA,
        }
    )

    # NET GAIN
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="true"
    )
    metrics_dict.update({MetricsNames.CONFUSION_MATRIX_NORM.value: cm.copy()})
    NGR = cm[1, 1]
    NGC = cm[2, 2]
    metrics_dict.update(
        {
            MetricsNames.NET_GAIN_REFUTATION.value: NGR,
            MetricsNames.NET_GAIN_CONFIRMATION.value: NGC,
        }
    )

    # QUALITY
    non_ambiguous_idx = mts_gt != DecisionLabels.UNSURE.value
    ia_decision = ia_decision[non_ambiguous_idx]
    mts_gt = mts_gt[non_ambiguous_idx]
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="all"
    )
    final_true_positives = cm[2, 0] + cm[2, 2]  # Yu + Yc
    final_false_positives = cm[1, 2]  # Nc
    precision = final_true_positives / (
        final_true_positives + final_false_positives
    )  #  (Yu + Yc) / (Yu + Yc + Nc)

    positives = cm[2, :].sum()
    recall = final_true_positives / positives  # (Yu + Yc) / (Yu + Yn + Yc)

    metrics_dict.update(
        {
            MetricsNames.PRECISION.value: precision,
            MetricsNames.RECALL.value: recall,
        }
    )

    return metrics_dict


def get_results_logs_str(metrics_dict: dict):
    """Format all metrics as a str for logging."""
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
    return results_logs
