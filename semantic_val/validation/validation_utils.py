from typing import List
import laspy
import numpy as np
from enum import Enum

import pandas as pd
from pytorch_lightning import seed_everything
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry.point import Point
from shapely.ops import unary_union
from sklearn.metrics import confusion_matrix

from semantic_val.utils import utils

log = utils.get_logger(__name__)


class ShapeFileCols(Enum):
    NUMBER_OF_CANDIDATE_BUILDINGS_POINT = "B_NUM_PTS"

    MTS_TRUE_POSITIVE_FRAC = "B_FRAC_TP"
    MTS_GROUND_TRUTH = "MTS_B_GT"  # -1: ambiguous, 0: no, 1: yes

    IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC = "B_FRAC_YES"
    MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC = "B_FRAC_NO"
    IA_AVERAGE_BUILDINGS_PROBA_FLOAT = "B_AVG_PRED"

    # Fields about decision of module to confirm (Yes) or refute (No) candidate
    IA_DECISION = "IA_B_VALID"  # -1: unsure, 0: no, 1: yes


class MetricsNames(Enum):
    # Constraints:
    CONFIRMATION_ACCURACY = "CA"
    REFUTATION_ACCURACY = "RA"
    # To maximize:
    LEVEL_OF_ACHIEVED_ACCURATE_CONFIRMATION = "P_AC"
    LEVEL_OF_ACHIEVED_ACCURATE_REFUTATION = "P_AR"
    SURFACE_OF_ACHIEVED_ACCURATE_ACTION = "P_A(C*R)"
    # Metainfo to evaluate absolute gain and what rests to validate post-module
    PROPORTION_OF_UNCERTAINTY = "P_U"
    PROPORTION_OF_CONFIRMATION = "P_C"
    PROPORTION_OF_REFUTATION = "P_R"
    PROPORTION_OF_ACTIONS_TAKEN = "P_(C+R)"


TRUE_POSITIVE_CODE = [19]
FALSE_POSITIVE_CODE = [20, 110, 112, 114, 115]

MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS = 3

# could be increased to demand high global proba for confirmation
PROBA_DECISION_THRESHOLD_FOR_CONFIRMATION = 0.5
# could be diminushed to demand low global proba fort refutation
PROBA_DECISION_THRESHOLD_FOR_REFUTATION = 0.5

DECISION_LABELS = ["unsure", "not-building", "building"]

# # TODO: check what 104 is
def load_geodf_of_candidate_building_points(
    las_filepath: str,
    crs="EPSG:2154",
) -> GeoDataFrame:
    """
    Load a las that went through correction and was predicted by trained model.
    Focus on points that were detected as building (keep_classes).
    """
    log.info("Loading LAS.")
    las = laspy.read(las_filepath)
    # TODO: uncomment to focus on predicted points only !
    # [WARNING: we should assert that all points have preds for production mode]
    # las.points = las.points[las["BuildingsHasPreds"]]

    candidate_building_points_idx = np.isin(
        las["classification"], TRUE_POSITIVE_CODE + FALSE_POSITIVE_CODE
    )
    las.points = las[candidate_building_points_idx]
    include_colnames = ["classification", "BuildingsProba"]
    data = np.array([las[colname] for colname in include_colnames]).transpose()
    lidar_df = pd.DataFrame(data, columns=include_colnames)

    log.info("Turning LAS into GeoDataFrame.")
    # TODO: this lst comprehension is slow and could be improved.
    # GDF does not accept a map object here.
    geometry = [Point(xy) for xy in zip(las.x, las.y)]
    las_gdf = GeoDataFrame(lidar_df, crs=crs, geometry=geometry)

    las_gdf["TruePositive"] = las_gdf["classification"].apply(
        lambda x: 1 * (x in TRUE_POSITIVE_CODE)
    )
    del las_gdf["classification"]
    return las_gdf


def get_unique_geometry_from_points(lidar_geodf):
    df = lidar_geodf.copy().buffer(0.25)
    return df.unary_union


def fill_holes(shape):
    shape = shape.buffer(0.80)
    shape = unary_union(shape)
    shape = shape.buffer(-0.80)
    return shape


def simplify_shape(shape):
    return shape.simplify(0.2, preserve_topology=False)


def vectorize_into_candidate_building_shapes(lidar_geodf):
    """
    From LAS with original classification, get candidate shapes
    Rules: >3m², holes filled, simplified geometry.
    """
    log.info("Vectorizing into candidate buildings.")
    union = get_unique_geometry_from_points(lidar_geodf)
    # TODO: maybe use maps here for  GeoDataFrame ? Or compose functions ?
    log.info("Filling vector holes.")
    shapes_no_holes = [fill_holes(shape) for shape in union]
    log.info("Simplifying shapes.")
    shapes_simplified = [simplify_shape(shape) for shape in shapes_no_holes]
    candidate_buildings = geopandas.GeoDataFrame(
        shapes_simplified, columns=["geometry"], crs="EPSG:2154"
    )
    log.info(f"Keeping shapes larger than {MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS}m².")

    candidate_buildings = candidate_buildings[
        candidate_buildings.area > MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS
    ]
    candidate_buildings = candidate_buildings.reset_index().rename(
        columns={"index": "shape_idx"}
    )
    return candidate_buildings


def agg_pts_info_by_shape(shapes_gdf: GeoDataFrame, lidar_gdf: GeoDataFrame):
    """Group the info and preds of candidate building points forming a candidate bulding shape."""
    gdf = lidar_gdf.sjoin(shapes_gdf, how="inner", predicate="within")
    gdf = gdf.groupby("shape_idx")[["BuildingsProba", "TruePositive"]].agg(
        lambda x: x.tolist()
    )
    return gdf


def derive_raw_shape_level_indicators(gdf):
    """Derive raw shape level info from ground truths and predictions."""
    gdf = set_num_pts_col(gdf)
    gdf = set_mean_proba_col(gdf)
    gdf = set_frac_confirmed_building_col(gdf)
    gdf = set_frac_refuted_building_col(gdf)
    gdf = set_frac_false_positive_col(gdf)
    gdf = set_MTS_ground_truth_flag(gdf)
    return gdf


def set_num_pts_col(gdf):
    gdf[ShapeFileCols.NUMBER_OF_CANDIDATE_BUILDINGS_POINT.value] = gdf.apply(
        lambda x: len(x["BuildingsProba"]), axis=1
    )
    return gdf


def set_mean_proba_col(gdf):
    gdf[ShapeFileCols.IA_AVERAGE_BUILDINGS_PROBA_FLOAT.value] = gdf.apply(
        lambda x: np.mean(x["BuildingsProba"]), axis=1
    )
    return gdf


def get_frac_confirmed_building_points_(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr >= PROBA_DECISION_THRESHOLD_FOR_CONFIRMATION) / len(arr)


def set_frac_confirmed_building_col(gdf):
    gdf[
        ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    ] = gdf.apply(lambda x: get_frac_confirmed_building_points_(x), axis=1)
    return gdf


def get_frac_refuted_building_points_(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr < PROBA_DECISION_THRESHOLD_FOR_REFUTATION) / len(arr)


def get_frac_MTS_true_positives_(row):
    true_positives = row["TruePositive"]
    return np.mean(true_positives)


def set_frac_false_positive_col(gdf):
    gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value] = gdf.apply(
        lambda x: get_frac_MTS_true_positives_(x), axis=1
    )
    return gdf


def set_frac_refuted_building_col(gdf):
    gdf[ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value] = gdf.apply(
        lambda x: get_frac_refuted_building_points_(x), axis=1
    )
    return gdf


def set_MTS_ground_truth_flag_(row):
    """Helper : Based on the fraction of confirmed building points, set the nature of the shape or declare an ambiguous case (flag = -1)"""
    FP_FRAC = 0.05
    TP_FRAC = 0.95
    tp = ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value
    if row[tp] >= TP_FRAC:
        return DECISION_LABELS[2]
    elif row[tp] < FP_FRAC:
        return DECISION_LABELS[1]
    else:
        return DECISION_LABELS[0]


def set_MTS_ground_truth_flag(gdf):
    """Set flags fr false positive / false positive / ambigous MTS ground truths."""
    mts_gt = ShapeFileCols.MTS_GROUND_TRUTH.value
    gdf[mts_gt] = gdf.apply(set_MTS_ground_truth_flag_, axis=1)
    return gdf


def make_decision_(
    row, validation_threshold: float = 0.7, refutation_threshold: float = 0.7
):
    """Helper: module decision based on fraction of confirmed/refuted points"""
    yes_frac = ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    no_frac = ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    if row[yes_frac] >= validation_threshold:
        return DECISION_LABELS[2]
    elif row[no_frac] >= refutation_threshold:
        return DECISION_LABELS[1]
    else:
        return DECISION_LABELS[0]


def make_decisions(
    gdf, validation_threshold: float = 0.7, refutation_threshold: float = 0.7
):
    """Add different flags to study the validation quality."""
    ia_decision = ShapeFileCols.IA_DECISION.value
    gdf[ia_decision] = gdf.apply(
        lambda x: make_decision_(
            x,
            validation_threshold=validation_threshold,
            refutation_threshold=refutation_threshold,
        ),
        axis=1,
    )
    return gdf


def evaluate_decisions(gdf: geopandas.GeoDataFrame):
    """Get dict of metrics to evaluate how good module decisions where in reference to ground truths."""
    mts_gt = ShapeFileCols.MTS_GROUND_TRUTH.value
    ia_decision = ShapeFileCols.IA_DECISION.value

    # Level of validation achieved (--> 1 is perfect validation)
    cm = confusion_matrix(
        gdf[mts_gt], gdf[ia_decision], labels=DECISION_LABELS, normalize="true"
    )
    PAR = cm[1, 1]
    PAC = cm[2, 2]
    PACxR = PAC * PAR

    # Accuracy of decision taken (--> 1 is each decision was the good one)
    cm = confusion_matrix(
        gdf[mts_gt], gdf[ia_decision], labels=DECISION_LABELS, normalize="pred"
    )
    RA = cm[1, 1]
    CA = cm[2, 2]

    # Proportion of each decision among total of candidates (PCR -> 1 is when we make a decision for each one.)
    cm = confusion_matrix(
        gdf[mts_gt], gdf[ia_decision], labels=DECISION_LABELS, normalize="all"
    )
    PU, PR, PC = cm.sum(axis=1)
    PCR = PC + PR

    metrics_dict = {
        MetricsNames.REFUTATION_ACCURACY.value: RA,
        MetricsNames.CONFIRMATION_ACCURACY.value: CA,
        MetricsNames.LEVEL_OF_ACHIEVED_ACCURATE_REFUTATION.value: PAR,
        MetricsNames.LEVEL_OF_ACHIEVED_ACCURATE_CONFIRMATION.value: PAC,
        MetricsNames.SURFACE_OF_ACHIEVED_ACCURATE_ACTION.value: PACxR,
        MetricsNames.PROPORTION_OF_UNCERTAINTY.value: PU,
        MetricsNames.PROPORTION_OF_REFUTATION.value: PR,
        MetricsNames.PROPORTION_OF_CONFIRMATION.value: PC,
        MetricsNames.PROPORTION_OF_ACTIONS_TAKEN.value: PCR,
    }

    return metrics_dict


def apply_constraint_and_sort(
    df: geopandas.GeoDataFrame,
    minimal_confirmation_accuracy_threshold: float = 0.98,
    minimal_refutation_accuracy_threshold: float = 0.98,
):
    """Filter out metrics that do not"""
    ca = MetricsNames.CONFIRMATION_ACCURACY.value
    ra = MetricsNames.REFUTATION_ACCURACY.value
    N_0 = len(df)
    df = df[df[ca] > minimal_confirmation_accuracy_threshold]
    N_1 = len(df)
    df = df[df[ra] > minimal_refutation_accuracy_threshold]
    N_2 = len(df)
    log.info(
        f"Applying JC (>{minimal_confirmation_accuracy_threshold}) and JD (>{minimal_refutation_accuracy_threshold}) constraint :"
        f"N = {N_0} --JC-> {N_1} --TD-> {N_2}"
    )
    df = df.sort_values(
        MetricsNames.SURFACE_OF_ACHIEVED_ACCURATE_ACTION.value, ascending=False
    )
    return df
