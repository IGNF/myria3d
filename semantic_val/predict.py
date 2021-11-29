import os
import hydra
import laspy
from omegaconf import DictConfig
from typing import List, Optional
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import LightningLoggerBase

from semantic_val.utils import utils
from semantic_val.datamodules.datasets.SemValBuildings202110 import LidarValDataset
from semantic_val.datamodules.processing import CustomCompose, SelectSubTile, DataHandler, ToTensor
from semantic_val.datamodules.processing import collate_fn, load_las_data

from semantic_val.inspection.utils import (
    DecisionLabels,
    ShapeFileCols,
    change_filepath_suffix,
    get_inspection_gdf,
    load_geodf_of_candidate_building_points,
    reset_classification,
    update_las_with_decisions,
)


log = utils.get_logger(__name__)


def get_val_transforms(subtile_width_meters) -> CustomCompose:
    """Create a transform composition for val phase."""
    selection = SelectSubTile(
        subtile_width_meters=subtile_width_meters, method="predefined"
    )


def predict(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    assert os.path.exists(config.trainer.resume_from_checkpoint)

    batch_size = config.datamodule.get("batch_size", 32)

    lasfiles_dir = config.datamodule.get("lasfiles_dir", None)
    src_las = config.datamodule.get("src_las", None)
    subtile_width_meters = config.datamodule.get("subtile_width_meters", 50)
    subtile_overlap = config.datamodule.get("subtile_overlap", 0)

    files_to_predict = [os.path.join(lasfiles_dir, src_las)]

    selection = SelectSubTile(subtile_width_meters=subtile_width_meters, method="predefined")

    predict_dataset = LidarValDataset(
        files_to_predict,
        loading_function=load_las_data,
        transform=selection,   
        target_transform=None,
        subtile_width_meters=subtile_width_meters,
        subtile_overlap=subtile_overlap,
    )

    predict_dataloader = DataLoader(
        dataset=predict_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn,
    )

    data_handler = DataHandler()

    for file_to_predict in files_to_predict:
        data_handler.load_new_las_for_preds(file_to_predict)

    model: LightningModule = hydra.utils.instantiate(config.model)

    for batch in predict_dataloader: # ça marche ça? itérateur qui fait bien ce qu'on veut?
        outputs = model(batch)
        data_handler.update_las_with_preds(outputs, "predict")

    log_path = os.getcwd()
    data_handler.preds_dirpath = os.path.join(log_path, "prediction_preds")
    os.makedirs(data_handler.preds_dirpath, exist_ok=True)
    data_handler.save_las_with_preds_and_close("predict")


    las = load_geodf_of_candidate_building_points(data_handler.output_path)
    gdf_inspection = get_inspection_gdf(
            las,
            min_frac_confirmation=config.inspection.min_frac_confirmation,
            min_frac_refutation=config.inspection.min_frac_refutation,
            min_confidence_confirmation=config.inspection.min_confidence_confirmation,
            min_confidence_refutation=config.inspection.min_confidence_refutation,
        )

    shp_path = os.path.join(os.getcwd(), "inspection_shapefiles/")
    os.makedirs(shp_path, exist_ok=True)
    shp_all_path = os.path.join(
        shp_path, config.inspection.inspection_shapefile_name.format(subset="all")
    )

    csv_path = change_filepath_suffix(shp_all_path, ".shp", ".csv")
    mode = "w" if not os.path.isfile(csv_path) else "a"

    keep = [item.value for item in ShapeFileCols] + ["geometry"]
    shp_decisions = gdf_inspection[keep]
    shp_decisions.to_file(shp_all_path, mode=mode)

    for decision in DecisionLabels:
        subset_path = os.path.join(
            shp_path,
            config.inspection.inspection_shapefile_name.format(
                subset=decision.value
            ),
        )
        shp_subset = shp_decisions[
            shp_decisions[ShapeFileCols.IA_DECISION.value] == decision.value
        ]
        if not shp_subset.empty:
            mode = "w" if not os.path.isfile(subset_path) else "a"
            header = True if mode == "w" else False
            shp_subset.to_file(subset_path, mode=mode)

    if config.inspection.update_las:
        log.info("Loading LAS agin to update candidate points.")
        las = laspy.read(data_handler.output_path)
        las.classification = reset_classification(las.classification)
        las = update_las_with_decisions(las, gdf_inspection)
        out_dir = os.path.dirname(shp_all_path)
        out_dir = os.path.join(out_dir, "las")
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.basename(data_handler.output_path)
        out_path = os.path.join(out_dir, out_name)
        las.write(out_path)
        log.info(f"Saved updated LAS to {out_path}")

    return 


    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)





    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    # TODO: recursive programming
    if config.trainer.resume_from_checkpoint:
        utils.update_config_with_hyperparams(config)

    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    model = model.load_from_checkpoint(trainer.resume_from_checkpoint)
    trainer.predict(model=model, datamodule=datamodule)

    return




