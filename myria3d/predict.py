from glob import glob
import os
import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from tqdm import tqdm

from myria3d.utils import utils
from myria3d.models.interpolation import Interpolator


log = utils.get_logger(__name__)


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

    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.load_from_checkpoint(config.predict.ckpt_path)
    device = utils.define_device_from_config_param(config.predict.gpus)
    model.to(device)
    model.eval()

    # TODO: Interpolator could be instantiated directly via hydra.
    itp = Interpolator(
        interpolation_k=config.predict.interpolation_k,
        classification_dict=datamodule.dataset_description.get("classification_dict"),
        probas_to_save=config.predict.probas_to_save,
    )

    for batch in tqdm(datamodule.predict_dataloader()):
        batch.to(device)
        logits = model.predict_step(batch)["logits"]
        itp.store_predictions(logits, batch.idx_in_original_cloud)

    out_f = itp.reduce_predictions_and_save(
        config.predict.src_las, config.predict.output_dir
    )
    return out_f


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    """This wrapper allows to specify a hydra configuration from command line.

    The config file logged during training should be used for prediction. It should
    be edited so that it does not rely on unnecessary environment variables (`oc.env:` prefix).
    Parameters in configuration group `predict` can be specified directly in the config file
    or overriden via CLI at runtime.

    This wrapper supports running predictions for all files specified by a glob pattern given
    by config parameter predict.src_las.

    """
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from myria3d.utils import utils
    from myria3d.predict import predict

    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    # Parameter predict.src_las can be a path or a glob pattern
    # e.g. /path/to/files_*.las
    src_las_iterable = glob(config.predict.src_las)

    if not src_las_iterable:
        raise FileNotFoundError(
            f"Globing pattern {config.predict.src_las} (param predict.src_las) did not return any file."
        )

    # Iterate over the files and predict.
    for config.predict.src_las in tqdm(src_las_iterable):
        predict(config)


if __name__ == "__main__":
    main()
