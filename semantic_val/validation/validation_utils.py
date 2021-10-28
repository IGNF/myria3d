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


TRUE_POSITIVE_CODE = [19]
FALSE_POSITIVE_CODE = [20, 110, 112, 114, 115]

MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS = 3

# could be increased to demand higher proba  of being building for confirmation
CONFIDENCE_THRESHOLD_FOR_CONFIRMATION = 0.5
# could be augmented to demand higher proba of not being building for refutation
CONFIDENCE_THRESHOLD_FOR_REFUTATION = 0.5

DECISION_LABELS = ["unsure", "not-building", "building"]


class ShapeFileCols(Enum):
    # Ground Truths: -1: ambiguous, 0: no, 1: yes
    MTS_GROUND_TRUTH = "MTS_B_GT"
    # Decision of module to confirm (Yes) or refute (No) candidate
    IA_DECISION = "IA_B_VALID"  # -1: unsure, 0: no, 1: yes

    MTS_TRUE_POSITIVE_FRAC = "B_FRAC_TP"
    IA_AVERAGE_BUILDINGS_PROBA_FLOAT = "B_AVG_PRED"

    IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC = "B_FRAC_YES"
    MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC = "B_FRAC_NO"

    NUMBER_OF_CANDIDATE_BUILDINGS_POINT = "B_NUM_PTS"


class MetricsNames(Enum):
    # Amount of each deicison
    PROPORTION_OF_UNCERTAINTY = "P_UNSURE"
    PROPORTION_OF_CONFIRMATION = "P_CONFIRM"
    PROPORTION_OF_REFUTATION = "P_REFUTE"

    # To maximize:
    PROPORTION_OF_AUTOMATED_DECISIONS = "P_AUTO"
    # Constraints:
    CONFIRMATION_ACCURACY = "A_CONFIRM"
    REFUTATION_ACCURACY = "A_REFUTE"
    # Metainfo to evaluate absolute gain and what rests to validate post-module
    NET_GAIN_CONFIRMATION = "NG_CONFIRM"
    NET_GAIN_REFUTATION = "NG_REFUTE"


def get_inspection_shapefile(
    las_filepath: str,
    confirmation_threshold: float = 0.05,
    refutation_threshold: float = 1.0,
):
    """From a predicted LAS, returns the inspection shapefile (or None if no candidate buildings)."""
    las_gdf = load_geodf_of_candidate_building_points(las_filepath)

    if len(las_gdf) == 0:
        log.info("/!\ Skipping tile with no candidate building points.")
        return None

    shapes_gdf = vectorize_into_candidate_building_shapes(las_gdf)
    log.info("Grouping points and deriving indicators by shape")
    comparison = agg_pts_info_by_shape(shapes_gdf, las_gdf)

    log.info("Derive raw shape level info from ground truths and predictions.")
    comparison = derive_raw_shape_level_indicators(comparison)

    log.info("Confirm or refute each candidate building if enough confidence.")
    comparison = make_decisions(
        comparison,
        confimation_threshold=confirmation_threshold,
        refutation_threshold=refutation_threshold,
    )

    df_out = shapes_gdf.join(comparison, on="shape_idx", how="left")
    keep = [item.value for item in ShapeFileCols] + ["geometry"]
    df_out = df_out[keep]
    return df_out


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
    """Derive raw shape level info from ground truths (TruePositive) and predictions (BuildingsProba)"""
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
    data = row["BuildingsProba"]
    proba_building = np.array(data)
    return np.sum(proba_building >= CONFIDENCE_THRESHOLD_FOR_CONFIRMATION) / len(
        proba_building
    )


def set_frac_confirmed_building_col(gdf):
    gdf[
        ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    ] = gdf.apply(lambda x: get_frac_confirmed_building_points_(x), axis=1)
    return gdf


def get_frac_refuted_building_points_(row):
    data = row["BuildingsProba"]
    proba_not_building = 1 - np.array(data)
    return np.sum(proba_not_building >= CONFIDENCE_THRESHOLD_FOR_REFUTATION) / len(
        proba_not_building
    )


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
    """Helper : Based on the fraction of confirmed building points, set the nature of the shape or declare an ambiguous case"""
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
    row, confirmation_threshold: float = 0.05, refutation_threshold: float = 1.0
):
    """Helper: module decision based on fraction of confirmed/refuted points"""
    yes_frac = ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    no_frac = ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    if row[yes_frac] >= confirmation_threshold:
        return DECISION_LABELS[2]
    elif row[no_frac] >= refutation_threshold:
        return DECISION_LABELS[1]
    else:
        return DECISION_LABELS[0]


def make_decisions(
    gdf, confimation_threshold: float = 0.05, refutation_threshold: float = 1.0
):
    """Add different flags to study the validation quality."""
    ia_decision = ShapeFileCols.IA_DECISION.value
    gdf[ia_decision] = gdf.apply(
        lambda row: make_decision_(
            row,
            confirmation_threshold=confimation_threshold,
            refutation_threshold=refutation_threshold,
        ),
        axis=1,
    )
    return gdf


def evaluate_decisions(gdf: geopandas.GeoDataFrame):
    """
    Get dict of metrics to evaluate how good module decisions were in reference to ground truths.
    Decisions : U=Unsure, C=Confirmation, R=Refutation
    Maximization criteria :
      Proportion of each decision among total of candidates.
      We want to maximize it. The max is not 1 since there are "ambiguous ground truth" cases.
    Constraints:
      Confirmation/Refutation Accuracy.
      Equéals 1 if each decision to confirm or refute was the right one.
    Net gain:
      Proportions of accurate C/R.
      Equals 1 if we either confirmed or refuted every candidate that could be, being unsure only
      for ambiguous groud truths)
    """
    mts_gt = gdf[ShapeFileCols.MTS_GROUND_TRUTH.value]
    ia_decision = gdf[ShapeFileCols.IA_DECISION.value]

    # CRITERIA
    cm = confusion_matrix(mts_gt, ia_decision, labels=DECISION_LABELS, normalize="all")
    PU, PR, PC = cm.sum(axis=0)
    # Proportion of decisions made among total (= 1 - PU)
    PAD = PC + PR

    # CONSTRAINTS
    cm = confusion_matrix(mts_gt, ia_decision, labels=DECISION_LABELS, normalize="pred")
    RA = cm[1, 1]
    CA = cm[2, 2]

    # NET GAIN
    cm = confusion_matrix(mts_gt, ia_decision, labels=DECISION_LABELS, normalize="true")
    NGR = cm[1, 1]
    NGC = cm[2, 2]

    metrics_dict = {
        MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value: PAD,
        MetricsNames.CONFIRMATION_ACCURACY.value: CA,
        MetricsNames.REFUTATION_ACCURACY.value: RA,
        MetricsNames.PROPORTION_OF_UNCERTAINTY.value: PU,
        MetricsNames.PROPORTION_OF_CONFIRMATION.value: PC,
        MetricsNames.PROPORTION_OF_REFUTATION.value: PR,
        MetricsNames.NET_GAIN_REFUTATION.value: NGR,
        MetricsNames.NET_GAIN_CONFIRMATION.value: NGC,
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
        f"Applying confirmation accuracy (>{minimal_confirmation_accuracy_threshold}) and refutation accuracy (>{minimal_refutation_accuracy_threshold}) constraint :"
        f"N = {N_0} --CA-> {N_1} --RA-> {N_2}"
    )
    df = df.sort_values(
        MetricsNames.PROPORTION_OF_AUTOMATED_DECISIONS.value, ascending=False
    )
    return df
