from typing import List
import laspy
import numpy as np

import pandas as pd
import pyproj
from pytorch_lightning import seed_everything
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry.point import Point

# Get the LAS files
def load_post_correction_shapefile(filepath: str) -> GeoDataFrame:
    """
    Load a shapefile whose shape are all the detected buildings.
    Later, this could be generated on the fly from the LAS with predictions directly.
    """
    gdf = geopandas.read_file(filepath)
    gdf = gdf.drop(columns=["num_classe", "_area"])
    gdf = gdf.reset_index().rename(columns={"index": "shape_index"})

    candidate_buildings_classes = [
        "bati_valide",
        "bati_auto",
        "surdetection_pont",
        "surdetection_veget",
        "surdetection_vehicule",
        "surdetection_autre",
    ]
    gdf = gdf[gdf.classe.isin(candidate_buildings_classes)]
    return gdf


# TODO: check what 104 is
def load_post_correction_predicted_las(
    las_filepath: str,
    crs: pyproj.CRS,
    keep_classes: List[int] = [6, 19, 20, 110, 112, 114, 115],
) -> GeoDataFrame:
    """
    Load a las that went through correction and was predicted by trained model.
    Focus on points that were detected as building (keep_classes).
    """
    las = laspy.read(las_filepath)
    candidate_building = np.isin(las["classification"], keep_classes)
    las.points = las[candidate_building]
    lidar_df = pd.DataFrame(las["BuildingsProba"], columns=["BuildingsProba"])
    # TODO: this lst comprehension is slow and could be improved.
    geometry = [Point(xy) for xy in zip(las.x, las.y)]
    las_gdf = GeoDataFrame(lidar_df, crs=crs, geometry=geometry)
    return las_gdf


def compare_classification_with_predictions(
    shapes_gdf: GeoDataFrame, las_gdf: GeoDataFrame
):
    """Compare the model predictions for points that are candidate building and within a bulding shape."""
    # Keep only points that are within a detected shape
    lidar_geodf_inside = las_gdf.sjoin(shapes_gdf, how="inner", predicate="within")
    # Aggregate confusion of points from the same shape in a list
    lidar_geodf_inside_lists = lidar_geodf_inside.groupby("shape_index")[
        "BuildingsProba"
    ].agg(lambda x: x.tolist())
    return lidar_geodf_inside_lists


def proportion_of_confirmed_building_points(shape_signals):
    # use a threshold that varies
    arr = np.array(shape_signals)
    arr = np.sum(arr >= 0.5) / len(arr)
    return arr
