from enum import Enum

import geopandas
import laspy
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry.point import Point
from shapely.ops import unary_union
from sklearn.metrics import confusion_matrix

from semantic_val.utils import utils
from semantic_val.callbacks.predictions_callbacks import ChannelNames

log = utils.get_logger(__name__)


MTS_TRUE_POSITIVE_CODE = [19]
MTS_FALSE_POSITIVE_CODE = [20, 110, 112, 114, 115]

POINT_BUFFER_FOR_UNION = 0.5
CLOSURE_BUFFER = 0.80
SIMPLIFICATION_TOLERANCE_METERS = 1
SIMPLIFICATION_PRESERVE_TOPOLOGY = True
MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS = 3
SHARED_CRS = "EPSG:2154"

CLASSIFICATION_CHANNEL_NAME = "classification"
POINT_IDX_COLNAME = "point_idx"
SHAPE_IDX_COLNAME = "shape_idx"
TRUE_POSITIVES_COLNAME = "TruePositive"


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
    # Metainfo to evaluate absolute gain and what is still to inspect
    NET_GAIN_CONFIRMATION = "NG_CONFIRM"
    NET_GAIN_REFUTATION = "NG_REFUTE"


class DecisionLabels(Enum):
    UNSURE = "unsure"  # 0
    NOT_BUILDING = "not-building"  # 1
    BUILDING = "building"  # 2


DECISION_LABELS_LIST = [l.value for l in DecisionLabels]


# Functions


def get_inspection_shapefile(
    points_gdf: laspy.LasData,
    min_frac_confirmation: float = 0.05,
    min_frac_refutation: float = 1.0,
    min_confidence_confirmation: float = 0.5,
    min_confidence_refutation: float = 0.5,
):
    """From a predicted LAS, returns 1) inspection geoDataFrame and 2) inspection csv with point-level info, for thresholds optimization."""

    shapes_gdf = vectorize(points_gdf)
    no_candidate_buildings_shape = len(shapes_gdf) == 0
    if no_candidate_buildings_shape:
        return None

    log.info("Group points by shape")
    points_gdf = agg_pts_info_by_shape(shapes_gdf, points_gdf)

    log.info("Derive shape-level indicators.")
    points_gdf = derive_shape_indicators(
        points_gdf,
        min_confidence_confirmation=min_confidence_confirmation,
        min_confidence_refutation=min_confidence_refutation,
    )

    log.info("Confirm or refute each candidate building if enough confidence.")
    points_gdf = make_decisions(
        points_gdf,
        min_frac_confirmation=min_frac_confirmation,
        min_frac_refutation=min_frac_refutation,
    )

    df_out = shapes_gdf.join(points_gdf, on=SHAPE_IDX_COLNAME, how="left")
    return df_out


# NOTE: this could be splited into
# 1) Load predicted LAS and focus on subselection : MTS_TRUE_POSITIVE_CODE + MTS_FALSE_POSITIVE_CODE by default
# and Turn it into a geodataframe
# 2) Set the TruePositive flag, only useful for post-correction files...
def load_geodf_of_candidate_building_points(
    las_filepath: str,
) -> GeoDataFrame:
    """
    Load a las that went through correction and was predicted by trained model.
    Focus on points that were detected as building (keep_classes).
    """
    log.info("Loading LAS.")
    las = laspy.read(las_filepath)

    candidate_building_points_idx = np.isin(
        las.classification, MTS_TRUE_POSITIVE_CODE + MTS_FALSE_POSITIVE_CODE
    )
    las.points = las[candidate_building_points_idx]

    no_candidate_building_points = len(las.points) == 0
    if no_candidate_building_points:
        return None

    # TODO: abstract classification somewhere as a constant
    keep_cols = [CLASSIFICATION_CHANNEL_NAME, ChannelNames.BuildingsProba.value]
    data = [candidate_building_points_idx] + [las[c] for c in keep_cols]
    data = np.array(data).transpose()
    columns = [POINT_IDX_COLNAME] + keep_cols
    lidar_df = pd.DataFrame(data, columns=columns)

    log.info("Turning LAS into GeoDataFrame.")
    # TODO: this lst comprehension is slow and could be improved.
    # GDF does not accept a map object here.
    geometry = [Point(xy) for xy in zip(las.x, las.y)]
    las_gdf = GeoDataFrame(lidar_df, crs=SHARED_CRS, geometry=geometry)

    las_gdf[TRUE_POSITIVES_COLNAME] = las_gdf.classification.apply(
        lambda x: 1 * (x in MTS_TRUE_POSITIVE_CODE)
    )
    # TODO: add POINT_IDX_COLNAME to get back to the las

    del las_gdf[CLASSIFICATION_CHANNEL_NAME]
    return las_gdf


def get_unique_geometry_from_points(lidar_geodf):
    """Buffer all points to merge thoses within 2 * buffer from one another."""
    df = lidar_geodf.copy().buffer(POINT_BUFFER_FOR_UNION)
    return df.unary_union


def close_holes(shape):
    """Closure operation to fill holes in shape"""
    shape = shape.buffer(CLOSURE_BUFFER)
    shape = unary_union(shape)
    shape = shape.buffer(-CLOSURE_BUFFER)
    return shape


def simplify_shape(shape):
    """Reduce the number of points that define a shape."""
    return shape.simplify(
        SIMPLIFICATION_TOLERANCE_METERS,
        preserve_topology=SIMPLIFICATION_PRESERVE_TOPOLOGY,
    )


def vectorize(lidar_geodf):
    """
    From LAS with original classification, get candidate shapes
    Rules: holes filled, simplified geometry, area>=3m².
    """
    log.info("Vectorizing into candidate buildings.")
    union = get_unique_geometry_from_points(lidar_geodf)
    log.info("Filling vector holes.")
    shapes_no_holes = [close_holes(shape) for shape in union]
    log.info("Simplifying shapes.")
    shapes_simplified = [simplify_shape(shape) for shape in shapes_no_holes]
    candidates_gdf = geopandas.GeoDataFrame(
        shapes_simplified, columns=["geometry"], crs=SHARED_CRS
    )
    log.info(f"Keeping shapes larger than {MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS}m².")
    candidates_gdf = candidates_gdf[
        candidates_gdf.area >= MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS
    ]
    candidates_gdf = candidates_gdf.rename_axis(SHAPE_IDX_COLNAME).reset_index()
    return candidates_gdf


def agg_pts_info_by_shape(shapes_gdf: GeoDataFrame, points_gdf: GeoDataFrame):
    """Group the info, preds, and original LAS idx of candidate building points forming a candidate bulding shape."""
    gdf = points_gdf.sjoin(shapes_gdf, how="inner", predicate="within")
    gdf = gdf.groupby(SHAPE_IDX_COLNAME)[
        [POINT_IDX_COLNAME, ChannelNames.BuildingsProba.value, TRUE_POSITIVES_COLNAME]
    ].agg(lambda x: x.tolist())
    return gdf


def derive_shape_indicators(
    gdf,
    min_confidence_confirmation: float = 0.5,
    min_confidence_refutation: float = 0.5,
):
    """Derive raw shape level info from ground truths (TruePositive) and predictions (BuildingsProba)"""
    # POINTS-LEVEL DECISIONS
    gdf = set_frac_confirmed_building_col(
        gdf, min_confidence_confirmation=min_confidence_confirmation
    )
    gdf = set_frac_refuted_building_col(
        gdf, min_confidence_refutation=min_confidence_refutation
    )
    # METAINFO
    gdf = set_num_pts_col(gdf)
    gdf = set_mean_proba_col(gdf)
    # GROUND TRUTHS INFO
    gdf = set_frac_false_positive_col(gdf)
    gdf = set_MTS_ground_truth_flag(gdf)
    return gdf


def set_num_pts_col(gdf):
    """Count number of point in shape."""
    gdf[ShapeFileCols.NUMBER_OF_CANDIDATE_BUILDINGS_POINT.value] = gdf.apply(
        lambda x: len(x[ChannelNames.BuildingsProba.value]), axis=1
    )
    return gdf


def set_mean_proba_col(gdf):
    """Average probability of being a building."""
    gdf[ShapeFileCols.IA_AVERAGE_BUILDINGS_PROBA_FLOAT.value] = gdf.apply(
        lambda x: np.mean(x[ChannelNames.BuildingsProba.value]), axis=1
    )
    return gdf


def _get_frac_confirmed_building_points(row, min_confidence_confirmation: float = 0.5):
    """Helper: Get fraction of building points in each shape that are confirmed to be building with enough confidence."""
    data = row[ChannelNames.BuildingsProba.value]
    proba_building = np.array(data)
    return np.sum(proba_building >= min_confidence_confirmation) / len(proba_building)


def set_frac_confirmed_building_col(gdf, min_confidence_confirmation: float = 0.5):
    """Set fraction of building points in each shape that are confirmed to be building with enough confidence."""
    gdf[
        ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    ] = gdf.apply(
        lambda x: _get_frac_confirmed_building_points(
            x,
            min_confidence_confirmation=min_confidence_confirmation,
        ),
        axis=1,
    )
    return gdf


def _get_frac_refuted_building_points(row, min_confidence_refutation: float = 0.5):
    """Helper: Get fraction of building points in each shape that are confirmed to be not-building with enough confidence."""
    data = row[ChannelNames.BuildingsProba.value]
    proba_not_building = 1 - np.array(data)
    return np.sum(proba_not_building >= min_confidence_refutation) / len(
        proba_not_building
    )


def set_frac_refuted_building_col(
    gdf,
    min_confidence_refutation: float = 0.5,
):
    """Set fraction of building points in each shape that are confirmed to be not-building with enough confidence."""
    gdf[ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value] = gdf.apply(
        lambda x: _get_frac_refuted_building_points(
            x, min_confidence_refutation=min_confidence_refutation
        ),
        axis=1,
    )
    return gdf


def _get_frac_MTS_true_positives(row):
    """Helper: Get fraction of candidate building points from initial classification that were actual building."""
    true_positives = row[TRUE_POSITIVES_COLNAME]
    return np.mean(true_positives)


def set_frac_false_positive_col(gdf):
    """Set fraction of candidate building points from initial classification that were actual building."""
    gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value] = gdf.apply(
        lambda x: _get_frac_MTS_true_positives(x), axis=1
    )
    return gdf


def _set_MTS_ground_truth_flag(row):
    """Helper : Based on the fraction of confirmed building points, set the nature of the shape or declare an ambiguous case"""
    FP_FRAC = 0.05
    TP_FRAC = 0.95
    tp = ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value
    if row[tp] >= TP_FRAC:
        return DecisionLabels.BUILDING.value
    elif row[tp] < FP_FRAC:
        return DecisionLabels.NOT_BUILDING.value
    else:
        return DecisionLabels.UNSURE.value


def set_MTS_ground_truth_flag(gdf):
    """Set flags fr false positive / false positive / ambigous MTS ground truths."""
    mts_gt = ShapeFileCols.MTS_GROUND_TRUTH.value
    gdf[mts_gt] = gdf.apply(_set_MTS_ground_truth_flag, axis=1)
    return gdf


def _make_decision(
    row, min_frac_confirmation: float = 0.05, min_frac_refutation: float = 1.0
):
    """Helper: module decision based on fraction of confirmed/refuted points"""
    yes_frac = ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    no_frac = ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    if row[yes_frac] >= min_frac_confirmation:
        return DecisionLabels.BUILDING.value
    elif row[no_frac] >= min_frac_refutation:
        return DecisionLabels.NOT_BUILDING.value
    else:
        return DecisionLabels.UNSURE.value


def make_decisions(
    gdf, min_frac_confirmation: float = 0.05, min_frac_refutation: float = 1.0
):
    """Confirm or refute candidate building shape based on fraction of confirmed/refuted points."""
    ia_decision = ShapeFileCols.IA_DECISION.value
    gdf[ia_decision] = gdf.apply(
        lambda row: _make_decision(
            row,
            min_frac_confirmation=min_frac_confirmation,
            min_frac_refutation=min_frac_refutation,
        ),
        axis=1,
    )
    return gdf


def evaluate_decisions(gdf: geopandas.GeoDataFrame):
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
    """
    mts_gt = gdf[ShapeFileCols.MTS_GROUND_TRUTH.value]
    ia_decision = gdf[ShapeFileCols.IA_DECISION.value]

    # CRITERIA
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="all"
    )
    PU, PR, PC = cm.sum(axis=0)
    # Proportion of decisions made among total (= 1 - PU)
    PAD = PC + PR

    # CONSTRAINTS
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="pred"
    )
    RA = cm[1, 1]
    CA = cm[2, 2]

    # NET GAIN
    cm = confusion_matrix(
        mts_gt, ia_decision, labels=DECISION_LABELS_LIST, normalize="true"
    )
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


def change_filepath_suffix(origin_filepath, origin_suffix: str, target_suffix: str):
    """Change a filepath suffix."""
    assert origin_filepath.endswith(".shp")
    target_filepath = origin_filepath.replace(origin_suffix, target_suffix)
    return target_filepath
