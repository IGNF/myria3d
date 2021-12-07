# TODO: rename validation_utils into "decision"

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
MTS_AUTO_DETECTED_CODE = 6
MTS_TRUE_POSITIVE_CODE_LIST = [19]
MTS_FALSE_POSITIVE_CODE_LIST = [20, 110, 112, 114, 115]
MTS_FALSE_NEGATIVE_CODE_LIST = [21]
LAST_ECHO_CODE = 104

# FINAL CLASSIFICATION CODES
DEFAULT_CODE = 1
CONFIRMED_BUILDING_CODE = 19
REFUTED_BUILDING_CODE = 20

CLUSTER_TOLERANCE = 0.5  # meters
CLUSTER_MIN_POINTS = 10
SHARED_CRS = "EPSG:2154"


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
    # Metainfo to evaluate absolute gain and what is still to inspect
    NET_GAIN_CONFIRMATION = "NG_CONFIRM"
    NET_GAIN_REFUTATION = "NG_REFUTE"


class DecisionLabels(Enum):
    UNSURE = "unsure"  # 0
    NOT_BUILDING = "not-building"  # 1
    BUILDING = "building"  # 2


DECISION_LABELS_LIST = [l.value for l in DecisionLabels]


def prepare_las_for_decision(
    input_filepath: str,
    input_bd_topo_shp: str,
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
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_filepath,
                "override_srs": SHARED_CRS,
                "nosrs": True,
            },
            {
                "type": "filters.cluster",
                "min_points": CLUSTER_MIN_POINTS,
                "tolerance": CLUSTER_TOLERANCE,  # meters
                "where": candidates_where,
            },
            {
                "type": "filters.ferry",
                "dimensions": f"=>{ChannelNames.BDTopoOverlay.value}",
            },
            {
                "column": "PREC_PLANI",
                "datasource": input_bd_topo_shp,
                "dimension": f"{ChannelNames.BDTopoOverlay.value}",
                "type": "filters.overlay",
            },
            {
                "type": "filters.assign",
                "value": f"{ChannelNames.BDTopoOverlay.value} = 1 WHERE {ChannelNames.BDTopoOverlay.value} > 0",
            },
            {
                "type": "writers.las",
                "filename": output_filepath,
                "forward": "all",  # keep all dimensions based on input format
                "extra_dims": "all",  # keep all extra dims as well
            },
        ]
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
    overlay_bools,
    min_confidence_confirmation: float = 0.6,
    min_frac_confirmation: float = 0.5,
    min_confidence_refutation: float = 0.6,
    min_frac_refutation: float = 0.8,
    min_overlay_confirmation: float = 0.95,
):
    """Confirm or refute candidate building shape based on fraction of confirmed/refuted points and
    on fraction of points overlayed by a building shape in a database."""
    confirmed = np.mean(probas >= min_confidence_confirmation) >= min_frac_confirmation
    if confirmed:
        return DecisionLabels.BUILDING.value
    refuted = np.mean((1 - probas) >= min_confidence_refutation) >= min_frac_refutation
    overlayed = np.mean(overlay_bools) >= min_overlay_confirmation
    if refuted:
        if overlayed:
            return DecisionLabels.UNSURE.value
        else:
            return DecisionLabels.NOT_BUILDING.value
    if overlayed:
        return DecisionLabels.BUILDING.value

    return DecisionLabels.UNSURE.value


def split_idx_by_dim(dim_array):
    """Returns a sequence of arrays of indices of elements sharing the same value in dim_array"""
    idx = np.argsort(dim_array)
    sorted_dim_array = dim_array[idx]
    group_idx = np.array_split(idx, np.where(np.diff(sorted_dim_array) != 0)[0] + 1)
    return group_idx


def update_las_with_decisions(las, params):
    """
    Update point cloud classification channel.
    Params is a dict-like object with optimized decision thresholds.
    """

    # 1) Set to default all candidats points not belonging to a group
    candidate_building_points_mask = (
        las[ChannelNames.Classification.value] == MTS_AUTO_DETECTED_CODE
    )
    las[ChannelNames.Classification.value][
        candidate_building_points_mask
    ] = DEFAULT_CODE

    # 2) Decide at the group-level
    split_idx = split_idx_by_dim(las[ChannelNames.ClusterID.value])
    split_idx = split_idx[1:]  # remove large group with ClusterID = 0
    for pts_idx in tqdm(split_idx, desc="Updating LAS."):
        pts = las.points[pts_idx]
        decision = make_group_decision(
            pts[ChannelNames.BuildingsProba.value],
            pts[ChannelNames.BDTopoOverlay.value],
            **params,
        )
        if decision == DecisionLabels.UNSURE.value:
            las[ChannelNames.Classification.value][pts_idx] = MTS_AUTO_DETECTED_CODE
        elif decision == DecisionLabels.BUILDING.value:
            las[ChannelNames.Classification.value][pts_idx] = CONFIRMED_BUILDING_CODE
        elif decision == DecisionLabels.NOT_BUILDING.value:
            las[ChannelNames.Classification.value][pts_idx] = REFUTED_BUILDING_CODE
        else:
            raise KeyError(f"Unexpected IA decision: {decision}")
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
      Proportion of each decision among total of candidates.
      We want to maximize it. The max is not 1 since there are "ambiguous ground truth" cases.
    Constraints:
      Confirmation/Refutation Accuracy.
      Equals 1 if no confirmation or refutation was a mistake (-> being unsure should not decrease accuracy)
    Net gain:
      Proportions of accurate C/R.
      Equals 1 if we either confirmed or refuted every candidate that could be, being unsure only
      for ambiguous groud truths)
    Quality
      Precision and Recall resulting from C/R, assuming perfect posterior decision for unsure predictions.
      Only candidate shapes with known ground truths are considered.
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
    final_positives = cm[2, 0] + cm[2, 2] + cm[1, 2]  # Yu + Yc + Nc
    final_false_positives = cm[1, 2]  # Yr
    precision = final_positives / (final_positives + final_false_positives)

    positives = cm[2, :].sum()
    final_true_positives = cm[2, 0] + cm[2, 2]  # Yu + Yc
    recall = final_true_positives / positives

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
