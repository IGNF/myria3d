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
    FALSE_POSITIVE_COL = "MTS_FP"
    FRAC_OF_CONFIRMED_BUILDINGS_AMONG_CANDIDATE = "B_CONFIRM"
    FRAC_OF_REFUTED_BUILDINGS_AMONG_CANDIDATE = "B_REFUTED"
    NUMBER_OF_CANDIDATE_BUILDINGS_POINT = "B_NUM_PTS"
    MEAN_BUILDINGS_PROBA = "B_AVG_PRED"


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

    las_gdf["FalsePositive"] = las_gdf["classification"].apply(
        lambda x: 1 * (x in FALSE_POSITIVE_CODE)
    )
    del las_gdf["classification"]
    return las_gdf


def get_unique_geometry_from_points(lidar_geodf):
    df = lidar_geodf.copy().buffer(0.25)
    return df.unary_union


def fill_holes(shape):
    shape = shape.buffer(0.75)
    shape = unary_union(shape)
    shape = shape.buffer(-0.75)
    return shape


def simplify_shape(shape):
    return shape.simplify(0.1, preserve_topology=False)


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
    points_within = lidar_gdf.sjoin(shapes_gdf, how="inner", predicate="within")
    groups = points_within.groupby("shape_idx")[
        ["BuildingsProba", "FalsePositive"]
    ].agg(lambda x: x.tolist())

    groups[ShapeFileCols.NUMBER_OF_CANDIDATE_BUILDINGS_POINT.value] = groups.apply(
        lambda x: len(x["BuildingsProba"]), axis=1
    )
    groups[ShapeFileCols.MEAN_BUILDINGS_PROBA.value] = groups.apply(
        lambda x: np.mean(x["BuildingsProba"]), axis=1
    )
    groups[
        ShapeFileCols.FRAC_OF_CONFIRMED_BUILDINGS_AMONG_CANDIDATE.value
    ] = groups.apply(lambda x: get_frac_of_confirmed_building_points(x), axis=1)
    groups[
        ShapeFileCols.FRAC_OF_REFUTED_BUILDINGS_AMONG_CANDIDATE.value
    ] = groups.apply(lambda x: get_frac_of_refuted_building_points(x), axis=1)
    groups[ShapeFileCols.FALSE_POSITIVE_COL.value] = groups.apply(
        lambda x: get_frac_of_MTS_false_positives(x), axis=1
    )

    return groups


# TODO : use a threshold that varies
def get_frac_of_confirmed_building_points(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr >= 0.5) / len(arr)


def get_frac_of_refuted_building_points(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr <= 0.5) / len(arr)


def get_frac_of_MTS_false_positives(row):
    proba = row["FalsePositive"]
    arr = np.array(proba)
    return np.mean(arr)
