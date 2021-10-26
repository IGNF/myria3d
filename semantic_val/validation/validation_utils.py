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

from semantic_val.utils import utils

log = utils.get_logger(__name__)


class ShapeFileCols(Enum):
    NUMBER_OF_CANDIDATE_BUILDINGS_POINT = "B_NUM_PTS"

    MTS_TRUE_POSITIVE_FRAC = "B_FRAC_TP"
    MTS_TRUE_POSITIVE_FLAG = "B_MTS_TP"
    MTS_FALSE_POSITIVE_FLAG = "B_MTS_FP"
    MTS_AMBIGUOUS_CASE_FLAG = "B_MTS_AMBI"

    IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC = "B_FRAC_YES"
    MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC = "B_FRAC_NO"
    IA_AVERAGE_BUILDINGS_PROBA_FLOAT = "B_AVG_PRED"

    # Fields about decision of module to confirm (Yes) or refute (No) candidate
    IA_NATURE_OF_FINAL_DECISION_ENUM = "IA_B_VALID"  # -1: unsure, 0: no, 1: yes
    IA_CONFIRMED_BUILDING_FLAG = "IA_B_YES"
    IA_REFUTED_BUILDING_FLAG = "IA_B_NO"
    IA_IS_UNCERTAIN_FLAG = "IA_B_UNSUR"

    # Fields about the accuracy of the module decision
    IA_ACCURATE_CONFIRMATION_FLAG = "IA_ACC_YES"
    IA_ACCURATE_REFUTATION_FLAG = "IA_ACC_NO"

    IA_INACCURATE_CONFIRMATION_FLAG = "IA_BAD_YES"
    IA_INACCURATE_REFUTATION_FLAG = "IA_BAD_NO"


TRUE_POSITIVE_CODE = [19]
FALSE_POSITIVE_CODE = [20, 110, 112, 114, 115]

MINIMAL_AREA_FOR_CANDIDATE_BUILDINGS = 3

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
    # DEBUG :
    # las.points = las.points[:30000]
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


def compare_classification_with_predictions(
    shapes_gdf: GeoDataFrame, lidar_gdf: GeoDataFrame
):
    """Group the info and preds of candidate building points forming a candidate bulding shape."""

    gdf = lidar_gdf.sjoin(shapes_gdf, how="inner", predicate="within")
    gdf = gdf.groupby("shape_idx")[["BuildingsProba", "TruePositive"]].agg(
        lambda x: x.tolist()
    )
    gdf = set_num_pts_col(gdf)
    gdf = set_mean_proba_col(gdf)
    gdf = set_frac_confirmed_building_col(gdf)
    gdf = set_frac_refuted_building_col(gdf)
    gdf = set_frac_false_positive_col(gdf)
    gdf = set_MTS_ground_truth_flag(gdf)

    return gdf


def set_MTS_ground_truth_flag(gdf):
    """Set flags fr false positive / false positive / ambigous MTS ground truths."""
    FP_FRAC = 0.05
    TP_FRAC = 0.95
    gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FLAG.value] = (
        gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value] > TP_FRAC
    )
    gdf[ShapeFileCols.MTS_FALSE_POSITIVE_FLAG.value] = (
        gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value] < FP_FRAC
    )
    gdf[ShapeFileCols.MTS_AMBIGUOUS_CASE_FLAG.value] = 1 * gdf[
        ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value
    ].between(FP_FRAC, TP_FRAC, inclusive="both")
    return gdf


def set_frac_false_positive_col(gdf):
    gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FRAC.value] = gdf.apply(
        lambda x: get_frac_MTS_true_positives(x), axis=1
    )
    return gdf


def set_frac_refuted_building_col(gdf):
    gdf[ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value] = gdf.apply(
        lambda x: get_frac_refuted_building_points(x), axis=1
    )
    return gdf


def set_frac_confirmed_building_col(gdf):
    gdf[
        ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value
    ] = gdf.apply(lambda x: get_frac_confirmed_building_points(x), axis=1)
    return gdf


def set_mean_proba_col(gdf):
    gdf[ShapeFileCols.IA_AVERAGE_BUILDINGS_PROBA_FLOAT.value] = gdf.apply(
        lambda x: np.mean(x["BuildingsProba"]), axis=1
    )
    return gdf


def set_num_pts_col(gdf):
    gdf[ShapeFileCols.NUMBER_OF_CANDIDATE_BUILDINGS_POINT.value] = gdf.apply(
        lambda x: len(x["BuildingsProba"]), axis=1
    )
    return gdf


# TODO : use a threshold that varies
def get_frac_confirmed_building_points(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr >= 0.5) / len(arr)


def get_frac_refuted_building_points(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr <= 0.5) / len(arr)


def get_frac_MTS_true_positives(row):
    true_positives = row["TruePositive"]
    return np.mean(true_positives)


def make_decisions(
    gdf, validation_threshold: float = 0.7, refutation_threshold: float = 0.7
):
    """Add different flags to study the validation quality."""
    # DEBUG only : delete this bc should be done in shapefile creation

    gdf[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value] = 1 * (
        gdf[ShapeFileCols.IA_CONFIRMED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value]
        >= validation_threshold
    )
    gdf[ShapeFileCols.IA_REFUTED_BUILDING_FLAG.value] = 1 * (
        gdf[ShapeFileCols.MTS_REFUTED_BUILDINGS_AMONG_MTS_CANDIDATE_FRAC.value]
        >= refutation_threshold
    )
    gdf[ShapeFileCols.IA_IS_UNCERTAIN_FLAG.value] = 1 * (
        (
            gdf[ShapeFileCols.IA_REFUTED_BUILDING_FLAG.value]
            + gdf[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value]
        )
        == 0
    )
    gdf[ShapeFileCols.IA_NATURE_OF_FINAL_DECISION_ENUM.value] = -1 * (
        gdf[ShapeFileCols.IA_IS_UNCERTAIN_FLAG.value] == 1
    ) + 1 * (gdf[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value] == 1)

    return gdf


def set_inaccuracy_and_accuracy_flags(gdf):
    gdf[ShapeFileCols.IA_ACCURATE_CONFIRMATION_FLAG.value] = 1 * (
        (gdf[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value] == 1)
        & (gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FLAG.value] == 1)
        & (gdf[ShapeFileCols.MTS_AMBIGUOUS_CASE_FLAG.value] == 0)
    )
    gdf[ShapeFileCols.IA_ACCURATE_REFUTATION_FLAG.value] = 1 * (
        (gdf[ShapeFileCols.IA_REFUTED_BUILDING_FLAG.value] == 1)
        & (gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FLAG.value] == 0)
        & (gdf[ShapeFileCols.MTS_AMBIGUOUS_CASE_FLAG.value] == 0)
    )
    gdf[ShapeFileCols.IA_INACCURATE_CONFIRMATION_FLAG.value] = 1 * (
        (
            (gdf[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value] == 1)
            & (gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FLAG.value] == 0)
            & (gdf[ShapeFileCols.MTS_AMBIGUOUS_CASE_FLAG.value] == 0)
        )
        | (
            (gdf[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value] == 1)
            & (gdf[ShapeFileCols.MTS_AMBIGUOUS_CASE_FLAG.value] == 1)
        )
    )
    gdf[ShapeFileCols.IA_INACCURATE_REFUTATION_FLAG.value] = 1 * (
        (
            (gdf[ShapeFileCols.IA_REFUTED_BUILDING_FLAG.value] == 1)
            & (gdf[ShapeFileCols.MTS_TRUE_POSITIVE_FLAG.value] == 1)
        )
        | (
            (gdf[ShapeFileCols.IA_REFUTED_BUILDING_FLAG.value] == 1)
            & (gdf[ShapeFileCols.MTS_AMBIGUOUS_CASE_FLAG.value] == 1)
        )
    )
    return gdf


def evaluate_decisions(gdf: geopandas.GeoDataFrame):
    """Get dict of metrics to evaluate how good module decisions where in reference to ground truths."""
    metrics_dict = dict()

    gdf = set_inaccuracy_and_accuracy_flags(gdf)

    N_shapes_in_gdf = len(gdf)
    gdf_sum = gdf.sum()

    num_mts_true_positives = gdf_sum[ShapeFileCols.MTS_TRUE_POSITIVE_FLAG.value]
    # TODO: calculate and ignore false positives here...
    num_mts_false_positives = N_shapes_in_gdf - num_mts_true_positives

    num_confirmations = gdf_sum[ShapeFileCols.IA_CONFIRMED_BUILDING_FLAG.value]
    num_refutations = gdf_sum[ShapeFileCols.IA_REFUTED_BUILDING_FLAG.value]

    num_confirmations_accurate = gdf_sum[
        ShapeFileCols.IA_ACCURATE_CONFIRMATION_FLAG.value
    ]
    num_confirmations_inacurrate = gdf_sum[
        ShapeFileCols.IA_INACCURATE_CONFIRMATION_FLAG.value
    ]
    num_refutations_accurate = gdf_sum[ShapeFileCols.IA_ACCURATE_REFUTATION_FLAG.value]
    num_refutations_inaccurate = gdf_sum[
        ShapeFileCols.IA_INACCURATE_REFUTATION_FLAG.value
    ]

    metrics_dict.update({"JC": num_confirmations_accurate / num_confirmations})
    metrics_dict.update({"JD": num_refutations_accurate / num_refutations})

    metrics_dict.update({"TC_rel": num_confirmations / num_mts_true_positives})
    metrics_dict.update({"TD_rel": num_refutations / num_mts_false_positives})
    metrics_dict.update({"TC_abs": num_confirmations / N_shapes_in_gdf})
    metrics_dict.update({"TD_abs": num_refutations / N_shapes_in_gdf})

    # errors: rel: among feasible errors, which did we make | abs : global ratio of errors occuring
    metrics_dict.update(
        {"TCE_rel": num_confirmations_inacurrate / num_mts_false_positives}
    )
    metrics_dict.update(
        {"TDE_rel": num_refutations_inaccurate / num_mts_true_positives}
    )
    metrics_dict.update({"TCE_abs": num_confirmations_inacurrate / N_shapes_in_gdf})
    metrics_dict.update({"TDE_abs": num_refutations_inaccurate / N_shapes_in_gdf})

    metrics_dict.update({"TCJ": num_confirmations_accurate / num_mts_true_positives})
    metrics_dict.update({"TDJ": num_refutations_accurate / num_mts_false_positives})

    metrics_dict.update({"TCJxTDJ": metrics_dict["TCJ"] * metrics_dict["TDJ"]})
    return metrics_dict


def apply_constraint(
    gdf: geopandas.GeoDataFrame,
    minimal_JC_threshold: float = 0.98,
    minimal_JD_threshold: float = 0.98,
):
    """Filter out metrics that do not"""
    N_0 = len(gdf)
    gdf = gdf[gdf["JC"] > minimal_JC_threshold]
    N_1 = len(gdf)
    gdf = gdf[gdf["JD"] > minimal_JD_threshold]
    N_2 = len(gdf)
    log.info(
        f"Applying JC (>{minimal_JC_threshold}) and JD (>{minimal_JD_threshold}) constraint : N = {N_0} --JC-> {N_1} --TD-> {N_2}"
    )
    return gdf
