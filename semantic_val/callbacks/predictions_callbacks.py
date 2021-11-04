from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
import os

import laspy
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
import os.path as osp
import torch
from torch_geometric.nn.pool import knn

from semantic_val.utils import utils

log = utils.get_logger(__name__)


class ChannelNames(Enum):
    BuildingsPreds = "BuildingsPreds"
    BuildingsProba = "BuildingsProba"
    BuildingsConfusion = "BuildingsConfusion"
    BuildingsHasPredictions = "BuildingsHasPredictions"


class SavePreds(Callback):
    """
    A Callback to save predictions back to original LAS file.
    Keep a full LAS tile in memory until it changes and thus must be saved.
    Added channels: BuildingPreds, BuildingProba, BuildingConfusion
    """

    def __init__(
        self,
        save_predictions: bool = False,
        save_train_predictions_every_n_step: int = 10 ** 6,
    ):
        self.in_memory_tile_filepath = ""
        self.save_predictions = save_predictions
        self.save_train_predictions_every_n_step = save_train_predictions_every_n_step

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """Setup logging functionnalities ; create the outputs dir."""

        self.experiment = trainer.logger.experiment[0]
        log_path = os.getcwd()
        log.info(f"Saving results and logs to {log_path}")
        self.experiment.log_parameter("experiment_logs_dirpath", log_path)

        if self.save_predictions:
            self.preds_dirpath = osp.join(log_path, "validation_preds")
            os.makedirs(self.preds_dirpath, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # TODO: can we get rid of should_save_preds but deal with saving with other params?
        if self.save_predictions:
            reached_train_saving_step = (
                trainer.global_step % self.save_train_predictions_every_n_step == 0
            )
            if reached_train_saving_step:
                self.update_las_with_preds(outputs, "train")
                log.debug(
                    f"Saving train preds to disk for batch number {trainer.global_step}"
                )
                self.save_las_with_preds_and_close("train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_predictions:
            if outputs is not None:
                self.update_las_with_preds(outputs, "val")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.save_predictions:
            log.debug(
                f"Saving validation preds to disk after train step {trainer.global_step}.\n"
            )
            self.save_las_with_preds_and_close("val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_predictions:
            log.debug(
                f"Saving test preds to disk after train step {trainer.global_step}"
            )
            self.update_las_with_preds(outputs, "test")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.save_predictions:
            self.save_las_with_preds_and_close("test")

    def update_las_with_preds(self, outputs: dict, phase: str):
        """
        Save the predicted classes in las format with position. Load the las if necessary.

        :param outputs: outputs of a step.
        :param phase: train, val or test phase (str).
        """
        proba = outputs["proba"].detach()
        preds = outputs["preds"].detach()
        batch = outputs["batch"].detach()
        targets = outputs["targets"].detach()

        # Group idx and their associated filepath if they belong to same tile
        filepath_elem_idx_lists = {}
        for elem_idx in range(batch.batch_size):
            filepath = batch.filepath[elem_idx]
            if filepath not in filepath_elem_idx_lists:
                filepath_elem_idx_lists[filepath] = [elem_idx]
            else:
                filepath_elem_idx_lists[filepath].append(elem_idx)
        # assign by group of elements of the same tile.
        for filepath, elem_idx_list in filepath_elem_idx_lists.items():
            is_a_new_tile = self.in_memory_tile_filepath != filepath
            if is_a_new_tile:
                close_previous_las_first = self.in_memory_tile_filepath != ""
                if close_previous_las_first:
                    self.save_las_with_preds_and_close(phase)
                self.load_new_las_for_preds(filepath)
            with torch.no_grad():
                self.assign_outputs_to_tile(batch, elem_idx_list, preds, proba, targets)

    def assign_outputs_to_tile(self, batch, elem_idx_list, preds, proba, targets):
        """Set the predicted elements in the current tile."""

        elem_points_idx = (batch.batch_y[..., None] == torch.Tensor(elem_idx_list)).any(
            -1
        )
        elem_pos = batch.pos_copy[elem_points_idx].cpu()
        elem_preds = preds[elem_points_idx].cpu()
        elem_proba = proba[elem_points_idx][:, 1].cpu()
        elem_targets = targets[elem_points_idx].cpu()

        assign_idx = knn(self.current_las_pos, elem_pos, k=1, num_workers=1)[1]

        self.current_las[ChannelNames.BuildingsHasPredictions.value][assign_idx] = 1
        self.current_las[ChannelNames.BuildingsPreds.value][assign_idx] = elem_preds
        self.current_las[ChannelNames.BuildingsProba.value][assign_idx] = elem_proba
        elem_preds_confusion = self.get_confusion(elem_preds, elem_targets)
        self.current_las[ChannelNames.BuildingsConfusion.value][
            assign_idx
        ] = elem_preds_confusion

    def get_confusion(self, elem_preds, elem_targets):
        """Get a confusion vector: TN=0, Tp=1, FN=2, FP=3 - Nodata or Nan is 4."""
        A = elem_preds * (elem_preds == elem_targets)
        B = (2 + elem_preds) * (elem_preds != elem_targets)
        elem_preds_confusion = A + B
        return elem_preds_confusion

    def load_new_las_for_preds(self, filepath):
        """Load a LAS and add necessary extradims."""

        self.current_las = laspy.read(filepath)
        self.in_memory_tile_filepath = filepath

        coln = ChannelNames.BuildingsHasPredictions.value
        param = laspy.ExtraBytesParams(name=coln, type=int)
        self.current_las.add_extra_dim(param)
        self.current_las[coln][:] = 0

        coln = ChannelNames.BuildingsPreds.value
        param = laspy.ExtraBytesParams(name=coln, type=int)
        self.current_las.add_extra_dim(param)
        self.current_las[coln][:] = 0

        coln = ChannelNames.BuildingsProba.value
        param = laspy.ExtraBytesParams(name=coln, type=float)
        self.current_las.add_extra_dim(param)
        self.current_las[coln][:] = 0.0

        coln = ChannelNames.BuildingsConfusion.value
        param = laspy.ExtraBytesParams(name=coln, type=int)
        self.current_las.add_extra_dim(param)
        self.current_las[coln][:] = 0

        self.current_las_pos = np.asarray(
            [
                self.current_las.x,
                self.current_las.y,
                self.current_las.z,
            ],
            dtype=np.float32,
        ).transpose()
        self.current_las_pos = torch.from_numpy(
            self.current_las_pos,
        )

    def save_las_with_preds_and_close(self, phase):
        """After inference of classification in self.las_with_predictions, save:
        - The LAS with updated classification
        - A GeoTIFF of the classification, which is logged into Comet as well.
        """
        tile_id = Path(self.in_memory_tile_filepath).stem
        filename = f"{phase}_{tile_id}.las"
        output_path = osp.join(self.preds_dirpath, filename)
        self.current_las.write(output_path)
        log.info(f"Predictions saved to : {output_path}.")
        # Closing:
        self.in_memory_tile_filepath = ""
        del self.current_las
        del self.current_las_pos
