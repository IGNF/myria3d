import os
from typing import List, Optional
import laspy
import numpy as np

from omegaconf import DictConfig
import pandas as pd
import pyproj
from pytorch_lightning import seed_everything
import glob
import os.path as osp
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry.point import Point
from tqdm import tqdm
from semantic_val.utils import utils
from semantic_val.validation.validation_utils import (
    compare_classification_with_predictions,
    vectorize_into_candidate_building_shapes,
    load_geodf_of_candidate_building_points,
    # load_post_correction_shapefile,
    proportion_of_confirmed_building_points,
)

log = utils.get_logger(__name__)


# TODO: Use append mode in to_file mode=”a” to avoid ugly pandas concatenation
# TODO: Become independant of post-correction shapefile by creating shapefil "as if" it did not undergo correction ?
# We have Classification in the LAS files so we can evaluate method on the newly created shapefile.
# (we can identify surclassification on the fly with associated points).


def validate(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    contrasted_shapes = []
    input_las_dirpath = config.validation_module.predicted_las_dirpath
    las_filepath = glob.glob(osp.join(input_las_dirpath, "*.las"))
    for las_filepath in tqdm(las_filepath, desc="Evaluating predicted point cloud"):

        las_gdf = load_geodf_of_candidate_building_points(las_filepath)
        shapes_gdf = vectorize_into_candidate_building_shapes(las_gdf)
        # TODO: edit compare_classification_with_predictions, cf. load_post_correction_predicted_las
        contrasted_shape = compare_classification_with_predictions(shapes_gdf, las_gdf)
        contrasted_shapes.append(contrasted_shape)

    contrasted_shapes = pd.concat(contrasted_shapes)
    contrasted_shapes = contrasted_shapes.apply(
        lambda x: proportion_of_confirmed_building_points(x), axis=1
    )
    # join back with geometries
    df_out = shapes_gdf.join(contrasted_shapes, on="shape_index", how="left")

    log.info(f"Logging directory: {os.getcwd()}")
    output_shp = config.validation_module.operationnal_output_shapefile_name
    df_out.to_file(output_shp)
