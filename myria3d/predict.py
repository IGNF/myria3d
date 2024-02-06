import os
import os.path as osp
import sys

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from tqdm import tqdm

from myria3d.models.model import Model

sys.path.append(osp.dirname(osp.dirname(__file__)))
from myria3d.models.interpolation import Interpolator  # noqa
from myria3d.utils import utils  # noqa

log = utils.get_logger(__name__)


@utils.eval_time
def predict(config: DictConfig) -> str:
    """
    Inference pipeline.

    A lightning datamodule splits a single point cloud of arbitrary size (typically: 1km * 1km) into subtiles
    (typically 50m * 50m), which are grouped into batches that are fed to a trained neural network embedded into a lightning Module.

    Predictions happen on a subsampled version of each subtile, which needs to be propagated back to the complete
    point cloud via an Interpolator. This Interpolator also includes the creation of a new LAS file with additional
    dimensions, including predicted classification, entropy, and (optionnaly) predicted probability for each class.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        str: path to ouptut LAS.

    """

    # Those are the 2 needed inputs, in addition to the hydra config.
    assert os.path.exists(config.predict.ckpt_path)
    assert os.path.exists(config.predict.src_las)

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data(config.predict.src_las)

    # Do not require gradient for faster predictions
    torch.set_grad_enabled(False)
    model = Model.load_from_checkpoint(config.predict.ckpt_path)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    # TODO: Interpolator could be instantiated directly via hydra.
    itp = Interpolator(
        interpolation_k=config.predict.interpolator.interpolation_k,
        classification_dict=config.dataset_description.get("classification_dict"),
        probas_to_save=config.predict.interpolator.probas_to_save,
        predicted_classification_channel=config.predict.interpolator.get(
            "predicted_classification_channel", "PredictedClassification"
        ),
        entropy_channel=config.predict.interpolator.get("entropy_channel", "entropy"),
    )

    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)
        logits = model.predict_step(batch)["logits"]
        itp.store_predictions(logits, batch.idx_in_original_cloud)

    out_f = itp.reduce_predictions_and_save(
        config.predict.src_las, config.predict.output_dir, config.datamodule.get("epsg")
    )
    return out_f
