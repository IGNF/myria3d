import os
from typing import List, Optional

from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import glob
import os.path as osp
from tqdm import tqdm
from semantic_val.validation.validation_utils import (
    DecisionLabels,
    ShapeFileCols,
    change_filepath_suffix,
    get_inspection_gdf,
    load_geodf_of_candidate_building_points,
    update_las_with_decisions,
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
    inspection_shp_all_path = osp.join(
        os.getcwd(), config.inspection.inspection_shapefile_name.format("all")
    )
    inspection_shp_unsure_path = osp.join(
        os.getcwd(), config.inspection.inspection_shapefile_name.format("unsure")
    )
    pts_level_info_csv_path = change_filepath_suffix(
        inspection_shp_all_path, ".shp", ".csv"
    )
    for las_filepath in tqdm(las_filepath, desc="Evaluating predicted point cloud"):
        log.info(f"Evaluation of tile {las_filepath}...")
        points_gdf = load_geodf_of_candidate_building_points(las_filepath)

        if points_gdf is None:
            log.info("/!\ Skipping tile: there are no candidate building points.")
            continue

        gdf_inspection = get_inspection_gdf(
            points_gdf,
            min_frac_confirmation=config.inspection.min_frac_confirmation,
            min_frac_refutation=config.inspection.min_frac_refutation,
            min_confidence_confirmation=config.inspection.min_confidence_confirmation,
            min_confidence_refutation=config.inspection.min_confidence_refutation,
        )
        if gdf_inspection is None:
            log.info(
                f"No candidate shape could be derived from the N={len(points_gdf.points)} candidate points buildings."
            )
            continue

        mode = "w" if not osp.isfile(pts_level_info_csv_path) else "a"
        header = True if mode == "w" else False

        gdf_inspection.to_csv(
            pts_level_info_csv_path, mode=mode, index=False, header=header
        )

        keep = [item.value for item in ShapeFileCols] + ["geometry"]
        shp_inspection_all = gdf_inspection[keep]
        shp_inspection_all.to_file(
            inspection_shp_all_path, mode=mode, index=False, header=header
        )

        if config.inspection.update_las:
            points_gdf = update_las_with_decisions(points_gdf, gdf_inspection)
            out_dir = osp.dirname(config.inspection.comparison_shapefile_path)
            out_name = osp.basename(las_filepath)
            out_path = osp.join(out_dir, "las", out_name)
            points_gdf.write(out_path)

        shp_inspection_unsure = shp_inspection_all[
            shp_inspection_all[ShapeFileCols.IA_DECISION.value]
            == DecisionLabels.UNSURE.value
        ]
        shp_inspection_unsure.to_file(
            inspection_shp_unsure_path, mode=mode, index=False, header=header
        )
    log.info(f"Output inspection shapefile is {inspection_shp_unsure_path}")
