import os
import pickle
import hydra
import laspy
import torch
from omegaconf import DictConfig
from typing import Optional
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
)
from tqdm import tqdm

from semantic_val.decision.codes import reset_classification 
from semantic_val.utils import utils
from semantic_val.datamodules.processing import DataHandler

from semantic_val.decision.decide import (
    prepare_las_for_decision,
    update_las_with_decisions,
)


log = utils.get_logger(__name__)


@utils.eval_time
def predict(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Those are the 4 needed inputs
    assert os.path.exists(config.prediction.resume_from_checkpoint)
    assert os.path.exists(config.prediction.src_las)
    assert os.path.exists(config.prediction.best_trial_pickle_path)
    # Use of a pre-downloaded shapfile here is temporary/
    assert os.path.exists(config.optimize.input_bd_topo_shp)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_all_transforms()
    datamodule._set_predict_data([config.prediction.src_las])

    data_handler = DataHandler()
    data_handler.preds_dirpath = config.prediction.output_dir

    data_handler.load_las_for_proba_update(config.prediction.src_las)

    with torch.no_grad():
        model: LightningModule = hydra.utils.instantiate(config.model)
        model = model.load_from_checkpoint(config.prediction.resume_from_checkpoint)
        if "gpus" in config.trainer and config.trainer.gpus == 1:
            model.cuda()
        model.eval()

        for index, batch in tqdm(
            enumerate(datamodule.predict_dataloader()), desc="Batch inference..."
        ):
            if "gpus" in config.trainer and config.trainer.gpus == 1:
                batch.cuda()
            outputs = model.predict_step(batch)
            data_handler.update_las_with_proba(outputs, "predict")
            # if index > 2:
            #    break  ###### à supprimer ###################

    updated_las_path = data_handler.save_las_with_proba_and_close("predict")

    log.info("Prepare LAS...")
    prepare_las_for_decision(
        updated_las_path,
        config.optimize.input_bd_topo_shp,
        updated_las_path,
    )

    log.info("Update classification...")
    las = laspy.read(updated_las_path)
    with open(config.prediction.best_trial_pickle_path, "rb") as f:
        log.info(f"Using best trial from: {config.prediction.best_trial_pickle_path}")
        best_trial = pickle.load(f)
    # TODO: add a mts_auto_detected_code = XXX parameter in update_las_with_decision, if risk of change.

    las.classification = reset_classification(las.classification)
    las = update_las_with_decisions(las, best_trial.params)
    las.write(updated_las_path)
    log.info(f"Updated LAS saved to : {updated_las_path}")
