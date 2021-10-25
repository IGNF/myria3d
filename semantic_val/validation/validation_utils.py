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


class ShapeFileCols(Enum):
    NUMBER_OF_CANDIDATE_BUILDINGS_POINT = "B_NUM_PTS"

    FRAC_OF_MTS_TRUE_POSITIVE = "B_FRAC_TP"
    FLAG_MTS_TRUE_POSITIVE = "B_MTS_TP"

    FRAC_OF_CONFIRMED_BUILDINGS_AMONG_CANDIDATE = "B_FRAC_YES"
    FRAC_OF_REFUTED_BUILDINGS_AMONG_CANDIDATE = "B_FRAC_NO"
    MEAN_BUILDINGS_PROBA = "B_AVG_PRED"

    NATURE_OF_FINAL_DECISION = "IA_B_VALID"

    # subfield belows are equivalent to B_DECISION
    CONFIRMED_BUILDING = "IA_B_YES"
    REFUTED_BUILDING = "IA_B_NO"
    UNCERTAINTY = "IA_B_UNSUR"


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
    las = laspy.read(las_filepath)
    # TODO: uncomment to focus on predicted points only !
    # [WARNING: we should assert that all points have preds for production mode]
    # las.points = las.points[las["BuildingsHasPreds"]]

    candidate_building_points_idx = np.isin(
        las["classification"], TRUE_POSITIVE_CODE + FALSE_POSITIVE_CODE
    )
    las.points = las[candidate_building_points_idx]

    # DEBUG : for debug:
    # las.points = las.points[:50000]
    include_colnames = ["classification", "BuildingsProba"]
    data = np.array([las[colname] for colname in include_colnames]).transpose()
    lidar_df = pd.DataFrame(data, columns=include_colnames)
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
    union = get_unique_geometry_from_points(lidar_geodf)
    # TODO: maybe use maps here for  GeoDataFrame ? Or compose functions ?
    shapes_no_holes = [fill_holes(shape) for shape in union]
    shapes_simplified = [simplify_shape(shape) for shape in shapes_no_holes]
    candidate_buildings = geopandas.GeoDataFrame(
        shapes_simplified, columns=["geometry"], crs="EPSG:2154"
    )
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
    gdf = set_true_positive_col(gdf)

    return gdf


def validate(gdf, validation_threshold: float = 0.7, refutation_threshold: float = 0.7):
    """Add different flags to study the validation quality."""
    gdf[ShapeFileCols.CONFIRMED_BUILDING] = 1 * (
        gdf[ShapeFileCols.FRAC_OF_CONFIRMED_BUILDINGS_AMONG_CANDIDATE]
        > validation_threshold
    )
    gdf[ShapeFileCols.REFUTED_BUILDING] = 1 * (
        gdf[ShapeFileCols.FRAC_OF_REFUTED_BUILDINGS_AMONG_CANDIDATE]
        > refutation_threshold
    )
    gdf[ShapeFileCols.UNCERTAINTY] = 1 * (
        (gdf[ShapeFileCols.REFUTED_BUILDING] + gdf[ShapeFileCols.CONFIRMED_BUILDING])
        == 0
    )
    gdf[ShapeFileCols.NATURE_OF_FINAL_DECISION] = -1 * (
        gdf[ShapeFileCols.UNCERTAINTY] == 1
    ) + 1 * (gdf[ShapeFileCols.CONFIRMED_BUILDING] == 1)

    return gdf


def set_true_positive_col(gdf):
    gdf[ShapeFileCols.FLAG_MTS_TRUE_POSITIVE.value] = (
        gdf[ShapeFileCols.FRAC_OF_MTS_TRUE_POSITIVE.value] > 0.9
    )
    return gdf


def set_frac_false_positive_col(gdf):
    gdf[ShapeFileCols.FRAC_OF_MTS_TRUE_POSITIVE.value] = gdf.apply(
        lambda x: get_frac_MTS_true_positives(x), axis=1
    )
    return gdf


def set_frac_refuted_building_col(gdf):
    gdf[ShapeFileCols.FRAC_OF_REFUTED_BUILDINGS_AMONG_CANDIDATE.value] = gdf.apply(
        lambda x: get_frac_refuted_building_points(x), axis=1
    )
    return gdf


def set_frac_confirmed_building_col(gdf):
    gdf[ShapeFileCols.FRAC_OF_CONFIRMED_BUILDINGS_AMONG_CANDIDATE.value] = gdf.apply(
        lambda x: get_frac_confirmed_building_points(x), axis=1
    )
    return gdf


def set_mean_proba_col(gdf):
    gdf[ShapeFileCols.MEAN_BUILDINGS_PROBA.value] = gdf.apply(
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
