import os
import hydra
import torch
from omegaconf import DictConfig
from typing import Optional
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm

from src.utils import utils
from src.datamodules.processing import DataHandler


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
    assert os.path.exists(config.predict.resume_from_checkpoint)
    assert os.path.exists(config.predict.src_las)

    torch.set_grad_enabled(False)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_all_transforms()
    datamodule._set_predict_data([config.predict.src_las])

    # TODO: pass as a predict.data_handler config for full parameterization outside this repo scope.
    data_handler = DataHandler(
        config.predict.output_dir, classification_dict=datamodule.classification_dict
    )

    model: LightningModule = hydra.utils.instantiate(config.model, recursive=True)
    model = model.load_from_checkpoint(config.predict.resume_from_checkpoint)
    device = utils.define_device_from_config_param(config.trainer.gpus)
    model.to(device)
    model.eval()

    for index, batch in tqdm(
        enumerate(datamodule.predict_dataloader()), desc="Infering probabilities..."
    ):
        batch.to(device)
        outputs = model.predict_step(batch)
        data_handler.update_with_inference_outputs(outputs)
        # if index >= 1:
        #     break  ###### TODO - this is for debugging purposes ###################

    updated_las_path = data_handler._interpolate_classification_and_save("predict")
    log.info(f"Updated LAS saved to : {updated_las_path}")
