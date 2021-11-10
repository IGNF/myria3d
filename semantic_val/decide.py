import os
from typing import List, Optional
import laspy

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
    reset_classification,
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
    shp_path = osp.join(os.getcwd(), "inspection_shapefiles/")
    os.makedirs(shp_path, exist_ok=True)
    for las_filepath in tqdm(las_filepath, desc="Evaluating predicted point cloud"):
        log.info(f"Evaluation of tile {las_filepath}...")
        las = load_geodf_of_candidate_building_points(las_filepath)

        if las is None:
            log.info("/!\ Skipping tile: there are no candidate building points.")
            continue

        gdf_inspection = get_inspection_gdf(
            las,
            min_frac_confirmation=config.inspection.min_frac_confirmation,
            min_frac_refutation=config.inspection.min_frac_refutation,
            min_confidence_confirmation=config.inspection.min_confidence_confirmation,
            min_confidence_refutation=config.inspection.min_confidence_refutation,
        )
        if gdf_inspection is None:
            log.info(
                f"No candidate shape could be derived from the N={len(las.points)} candidate points buildings."
            )
            continue

        shp_all_path = osp.join(
            shp_path, config.inspection.inspection_shapefile_name.format(subset="all")
        )
        csv_path = change_filepath_suffix(shp_all_path, ".shp", ".csv")
        mode = "w" if not osp.isfile(csv_path) else "a"
        header = True if mode == "w" else False

        gdf_inspection.to_csv(csv_path, mode=mode, index=False, header=header)

        keep = [item.value for item in ShapeFileCols] + ["geometry"]
        shp_decisions = gdf_inspection[keep]
        shp_decisions.to_file(shp_all_path, mode=mode)

        for decision in DecisionLabels:
            subset_path = osp.join(
                shp_path,
                config.inspection.inspection_shapefile_name.format(
                    subset=decision.value
                ),
            )
            shp_subset = shp_decisions[
                shp_decisions[ShapeFileCols.IA_DECISION.value] == decision.value
            ]
            if not shp_subset.empty:
                mode = "w" if not osp.isfile(subset_path) else "a"
                header = True if mode == "w" else False
                shp_subset.to_file(subset_path, mode=mode)

        if config.inspection.update_las:
            log.info("Loading LAS agin to update candidate points.")
            las = laspy.read(las_filepath)
            las.classification = reset_classification(las.classification)
            las = update_las_with_decisions(las, gdf_inspection)
            out_dir = osp.dirname(shp_all_path)
            out_dir = osp.join(out_dir, "las")
            os.makedirs(out_dir, exist_ok=True)
            out_name = osp.basename(las_filepath)
            out_path = osp.join(out_dir, out_name)
            las.write(out_path)
            log.info(f"Saved updated LAS to {out_path}")

    log.info(f"Output inspection shapefile is {shp_all_path}")
