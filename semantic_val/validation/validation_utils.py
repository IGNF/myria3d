from typing import List
import laspy
import numpy as np

import pandas as pd
import pyproj
from pytorch_lightning import seed_everything
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry.point import Point
from shapely.ops import unary_union

# Get the LAS files
# def load_post_correction_shapefile(filepath: str) -> GeoDataFrame:
#     """
#     Load a shapefile whose shape are all the detected buildings.
#     Later, this could be generated on the fly from the LAS with predictions directly.
#     """
#     gdf = geopandas.read_file(filepath)
#     gdf = gdf.drop(columns=["num_classe", "_area"])
#     gdf = gdf.reset_index().rename(columns={"index": "shape_index"})

#     candidate_buildings_classes = [
#         "bati_valide",
#         "bati_auto",
#         "surdetection_pont",
#         "surdetection_veget",
#         "surdetection_vehicule",
#         "surdetection_autre",
#     ]
#     gdf = gdf[gdf.classe.isin(candidate_buildings_classes)]
#     return gdf


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

    true_positive_code = [19]
    false_positive_codes = [20, 110, 112, 114, 115]
    candidate_building = np.isin(
        las["classification"], true_positive_code + false_positive_codes
    )
    las.points = las[candidate_building]

    # for debug:
    # las.points = las.points[:50000]
    include_colnames = ["classification", "BuildingsProba"]
    data = np.array([las[colname] for colname in include_colnames]).transpose()
    lidar_df = pd.DataFrame(data, columns=include_colnames)
    # TODO: this lst comprehension is slow and could be improved.
    geometry = [Point(xy) for xy in zip(las.x, las.y)]
    las_gdf = GeoDataFrame(lidar_df, crs=crs, geometry=geometry)

    las_gdf["FalsePositive"] = las_gdf["classification"].apply(
        lambda x: 1 * (x in false_positive_codes)
    )
    return las_gdf


def get_unique_geometry_from_points(lidar_geodf):
    df = lidar_geodf.copy()
    df = df.buffer(0.25)
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
    shapes_no_holes = [fill_holes(shape) for shape in union]
    shapes_simplified = [simplify_shape(shape) for shape in shapes_no_holes]
    candidate_buildings = geopandas.GeoDataFrame(
        shapes_simplified, columns=["geometry"], crs="EPSG:2154"
    )
    MINIMAL_AREA = 3
    candidate_buildings = candidate_buildings[candidate_buildings.area > MINIMAL_AREA]
    candidate_buildings = candidate_buildings.reset_index().rename(
        columns={"index": "shape_index"}
    )
    return candidate_buildings


def compare_classification_with_predictions(
    shapes_gdf: GeoDataFrame, lidar_gdf: GeoDataFrame
):
    """Compare the model predictions for points that are candidate building and within a bulding shape."""
    # Keep only points that are within a detected shape
    lidar_geodf_inside = lidar_gdf.sjoin(shapes_gdf, how="inner", predicate="within")
    # Aggregate confusion of points from the same shape in a list
    # TODO: also aggregate FalsePositive flag.
    lidar_geodf_inside_lists = lidar_geodf_inside.groupby("shape_index")[
        ["BuildingsProba", "FalsePositive"]
    ].agg(lambda x: x.tolist())
    return lidar_geodf_inside_lists


# TODO : use a threshold that varies
def proportion_of_confirmed_building_points(row):
    proba = row["BuildingsProba"]
    arr = np.array(proba)
    return np.sum(arr >= 0.5) / len(arr)


def proportion_of_false_positives(row):
    proba = row["FalsePositive"]
    arr = np.array(proba)
    return np.mean(arr)
