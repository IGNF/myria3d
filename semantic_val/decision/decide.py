import geopandas

import os
import json
import re
from typing import List, Union

import numpy as np
import pdal
from tqdm import tqdm

from semantic_val.decision.codes import *
from semantic_val.datamodules.processing import ChannelNames
from semantic_val.utils import utils
from semantic_val.utils.db_communication import ConnectionData, db_communication

log = utils.get_logger(__name__)

# GLOBAL PARAMETERS
CLUSTER_TOLERANCE = 0.5  # meters
CLUSTER_MIN_POINTS = 10
SHARED_CRS_PREFIX = "EPSG:"
SHARED_CRS = 2154
SHAPEFILE_SUBDIR = "shp"
SHAPEFILE_NAME = "temp_shapefile.shp"

def prepare_las_for_decision(
    input_filepath: str,
    data_connexion_db: ConnectionData,
    output_filepath: str,
    candidate_building_points_classification_code: Union[int, List[int]] = [
        MTS_AUTO_DETECTED_CODE
    ]
    + MTS_TRUE_POSITIVE_CODE_LIST
    + MTS_FALSE_POSITIVE_CODE_LIST,
):
    """
    Prepare las for later decision process.
    Will:
    - Cluster candidates points, thus creating a ClusterId channel (default cluster: 0).
    - Identify points overlayed by a BDTopo shape, thus creating a BDTopoOverlay channel (no overlap: 0).
    """

    bd_topo_shp_dir_path = os.path.join(os.path.dirname(output_filepath), SHAPEFILE_SUBDIR)
    if os.path.exists(bd_topo_shp_dir_path):
        # remove any existing file in the directory
        for root, _, files in os.walk(bd_topo_shp_dir_path):
            for f in files:
                os.remove(os.path.join(root, f))
    else:
        os.mkdir(bd_topo_shp_dir_path)

    shapefile_path = os.path.join(bd_topo_shp_dir_path, SHAPEFILE_NAME)
    db_communication(
        data_connexion_db,
        *extract_coor(os.path.basename(input_filepath), 1000, 1000, 50),
        SHARED_CRS,
        shapefile_path
        )

    # a column with only "1"
    # gdf = geopandas.read_file(shapefile_path)
    # gdf["presence"] = 1
    # gdf[["presence","geometry"]].to_file(shapefile_path)

    if isinstance(candidate_building_points_classification_code, int):
        candidate_building_points_classification_code = [
            candidate_building_points_classification_code
        ]
    candidates_where = (
        "("
        + " || ".join(
            f"Classification == {int(candidat_code)}"
            for candidat_code in candidate_building_points_classification_code
        )
        + ")"
    )
    _reader = [
        {
            "type": "readers.las",
            "filename": input_filepath,
            "override_srs": SHARED_CRS_PREFIX + str(SHARED_CRS),
            "nosrs": True,
        }
    ]
    _cluster = [
        {
            "type": "filters.cluster",
            "min_points": CLUSTER_MIN_POINTS,
            "tolerance": CLUSTER_TOLERANCE,  # meters
            "where": candidates_where,
        }
    ]
    _topo_overlay = [
        {
            "type": "filters.ferry",
            "dimensions": f"=>{ChannelNames.BDTopoOverlay.value}",
        },
        {
            "column": "PRESENCE",
            "datasource": shapefile_path,
            "dimension": f"{ChannelNames.BDTopoOverlay.value}",
            "type": "filters.overlay",
        },
    ]
    _writer = [
        {
            "type": "writers.las",
            "filename": output_filepath,
            "forward": "all",  # keep all dimensions based on input format
            "extra_dims": "all",  # keep all extra dims as well
        }
    ]
    pipeline = {"pipeline": _reader + _cluster + _topo_overlay + _writer}
    pipeline = json.dumps(pipeline)
    pipeline = pdal.Pipeline(pipeline)
    pipeline.execute()
    structured_array = pipeline.arrays[0]
    return structured_array


def extract_coor(las_name: str, x_span: float, y_span: float, buffer: float):
    """extract the dimensions from the LAS name, the spans desired and a buffer"""
    x_min, y_min = re.findall(r'[0-9]{4,10}', las_name)   # get the values with [4,10] digits in the file name
    x_min, y_min = int(x_min), int(y_min)
    return x_min - buffer, y_min - buffer, x_min + x_span + buffer, y_min + y_span + buffer


def make_group_decision(*args, **kwargs):
    detailed_code = make_detailed_group_decision(*args, **kwargs)
    return DETAILED_CODE_TO_FINAL_CODE[detailed_code]


def make_detailed_group_decision(
    probas,
    topo_overlay_bools,
    min_confidence_confirmation: float = 0.6,
    min_frac_confirmation: float = 0.5,
    min_confidence_refutation: float = 0.6,
    min_frac_refutation: float = 0.8,
    min_overlay_confirmation: float = 0.95,
):
    """
    Confirm or refute candidate building shape based on fraction of confirmed/refuted points and
    on fraction of points overlayed by a building shape in a database.
    """
    ia_confirmed = (
        np.mean(probas >= min_confidence_confirmation) >= min_frac_confirmation
    )
    ia_refuted = (
        np.mean((1 - probas) >= min_confidence_refutation) >= min_frac_refutation
    )
    topo_overlayed = np.mean(topo_overlay_bools) >= min_overlay_confirmation

    if ia_refuted:
        if topo_overlayed:
            return DetailedClassificationCodes.IA_REFUTED_AND_DB_OVERLAYED.value
        return DetailedClassificationCodes.IA_REFUTED.value
    if ia_confirmed:
        if topo_overlayed:
            return DetailedClassificationCodes.BOTH_CONFIRMED.value
        return DetailedClassificationCodes.IA_CONFIRMED_ONLY.value
    if topo_overlayed:
        return DetailedClassificationCodes.DB_OVERLAYED_ONLY.value
    return DetailedClassificationCodes.BOTH_UNSURE.value


def update_las_with_decisions(
    las,
    params,
    use_final_classification_codes: bool = True,
    mts_auto_detected_code: int = MTS_AUTO_DETECTED_CODE,
):
    """
    Update point cloud classification channel.
    Params is a dict-like object with optimized decision thresholds.
    """

    # 1) Set to default all candidats points
    candidate_building_points_mask = (
        las[ChannelNames.Classification.value] == mts_auto_detected_code
    )
    las[ChannelNames.Classification.value][
        candidate_building_points_mask
    ] = DEFAULT_CODE

    # 2) Decide at the group-level
    split_idx = split_idx_by_dim(las[ChannelNames.ClusterID.value])
    split_idx = split_idx[1:]  # remove unclustered group with ClusterID = 0
    for pts_idx in tqdm(split_idx, desc="Updating LAS."):
        pts = las.points[pts_idx]
        detailed_code = make_detailed_group_decision(
            pts[ChannelNames.BuildingsProba.value],
            pts[ChannelNames.BDTopoOverlay.value],
            **params,
        )
        if use_final_classification_codes:
            las[ChannelNames.Classification.value][
                pts_idx
            ] = DETAILED_CODE_TO_FINAL_CODE[detailed_code]
        else:
            las[ChannelNames.Classification.value][pts_idx] = detailed_code

    return las


def split_idx_by_dim(dim_array):
    """Returns a sequence of arrays of indices of elements sharing the same value in dim_array"""
    idx = np.argsort(dim_array)
    sorted_dim_array = dim_array[idx]
    group_idx = np.array_split(idx, np.where(np.diff(sorted_dim_array) != 0)[0] + 1)
    return group_idx
