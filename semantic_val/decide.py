import os
from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import glob
import os.path as osp
from tqdm import tqdm
from semantic_val.validation.validation_utils import (
    ShapeFileCols,
    change_filepath_suffix,
    get_inspection_shapefile,
    load_geodf_of_candidate_building_points,
)
from semantic_val.utils import utils

log = utils.get_logger(__name__)


def decide(config: DictConfig) -> Optional[float]:
    """
    Contains decision pipeline, which takes predicted las and generate a single,
    large shapefile with vectorized candidate building and the trained model
    decision (confirm/refute). A csv with list of points-level info is also generated for HPO.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    log.info(f"Logging directory: {os.getcwd()}")

    las_filepath = glob.glob(osp.join(config.inspection.predicted_las_dirpath, "*.las"))
    inspection_shp_path = config.inspection.comparison_shapefile_path
    pts_level_info_csv_path = change_filepath_suffix(
        inspection_shp_path, ".shp", ".csv"
    )
    for las_filepath in tqdm(las_filepath, desc="Evaluating predicted point cloud"):
        log.info(f"Evaluation of tile {las_filepath}...")
        points_gdf = load_geodf_of_candidate_building_points(las_filepath)

        if points_gdf is None:
            log.info("/!\ Skipping tile: there are no candidate building points.")
            continue

        df_out = get_inspection_shapefile(
            points_gdf,
            min_frac_confirmation=config.inspection.min_frac_confirmation,
            min_frac_refutation=config.inspection.min_frac_refutation,
            min_confidence_confirmation=config.inspection.min_confidence_confirmation,
            min_confidence_refutation=config.inspection.min_confidence_refutation,
        )
        if df_out is None:
            log.info(
                f"No candidate shape could be derived from the N={len(points_gdf.points)} candidate points buildings."
            )
            continue

        mode = "w" if not osp.isfile(pts_level_info_csv_path) else "a"
        header = True if mode == "w" else False

        df_out.to_csv(pts_level_info_csv_path, mode=mode, index=False, header=header)

        keep = [item.value for item in ShapeFileCols] + ["geometry"]
        gdf_out = df_out[keep]
        gdf_out.to_file(inspection_shp_path, mode=mode, index=False, header=header)

        update_las_with_decisions(points_gdf, df_out)

    log.info(f"Output shapefile is in {inspection_shp_path}")
