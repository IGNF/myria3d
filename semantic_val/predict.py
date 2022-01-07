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

from semantic_val.utils.db_communication import ConnectionData
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

    # Those are the 3 needed inputs
    assert os.path.exists(config.prediction.resume_from_checkpoint)
    assert os.path.exists(config.prediction.src_las)
    assert os.path.exists(config.prediction.best_trial_pickle_path)

    torch.set_grad_enabled(False)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_all_transforms()
    datamodule._set_predict_data(
        [config.prediction.src_las], config.prediction.mts_auto_detected_code
    )

    data_handler = DataHandler(preds_dirpath=config.prediction.output_dir)
    data_handler.load_las_for_proba_update(config.prediction.src_las)

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.prediction.resume_from_checkpoint)
    device = utils.define_device_from_config_param(config.trainer.gpus)
    model.to(device)
    model.eval()

    for index, batch in tqdm(
        enumerate(datamodule.predict_dataloader()), desc="Infering probabilities..."
    ):
        batch.to(device)
        outputs = model.predict_step(batch)
        data_handler.append_pos_and_proba_to_list(outputs)
        # if index >= 1:
        #     break  ###### TODO - this is for debugging purposes ###################

    updated_las_path = data_handler.interpolate_probas_and_save("predict")

    data_connexion_db = ConnectionData(
        config.prediction.host,
        config.prediction.user,
        config.prediction.pwd,
        config.prediction.bd_name,
    )

    log.info("Prepare LAS...")
    prepare_las_for_decision(
        updated_las_path,
        data_connexion_db,
        updated_las_path,
        candidate_building_points_classification_code=[
            config.prediction.mts_auto_detected_code
        ],
    )

    log.info("Update classification...")
    las = laspy.read(updated_las_path)
    with open(config.prediction.best_trial_pickle_path, "rb") as f:
        log.info(f"Using best trial from: {config.prediction.best_trial_pickle_path}")
        best_trial = pickle.load(f)

    las = update_las_with_decisions(
        las,
        best_trial.params,
        use_final_classification_codes=config.prediction.use_final_classification_codes,
        mts_auto_detected_code=config.prediction.mts_auto_detected_code,
    )
    las.write(updated_las_path)
    log.info(f"Updated LAS saved to : {updated_las_path}")
