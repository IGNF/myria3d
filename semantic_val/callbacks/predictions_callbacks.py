import os
from pathlib import Path
from typing import Any, List, Optional
import laspy
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT

import torch
from comet_ml import Experiment
from torch_geometric.nn.pool import knn
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CometLogger, LoggerCollection
from pytorch_lightning.utilities import rank_zero_only
import pytorch_lightning as pl
from torch_geometric.data.batch import Batch
from semantic_val.utils import utils
import os.path as osp

log = utils.get_logger(__name__)


class SavePreds(Callback):
    """
    A Callback to save predictions back to original LAS file.
    Keep a full LAS tile in memory until it changes and thus must be saved.
    Added channels: BuildingPreds, BuildingProba, BuildingConfusion
    """

    def __init__(self):
        self.in_memory_tile_filepath = ""
        self.train_step_global_idx: int = 0

    def on_init_end(self, trainer: pl.Trainer) -> None:
        """Setup logging functionnalities ; create the outputs dir."""

        self.experiment = trainer.logger.experiment[0]
        log_path = os.getcwd()
        log.info(f"Saving results and logs to {log_path}")

        self.preds_dirpath = osp.join(log_path, "validation_preds")
        os.makedirs(self.preds_dirpath, exist_ok=True)

        self.experiment.log_parameter("experiment_logs_dirpath", log_path)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Keep track of global step idx."""

        self.train_step_global_idx = self.train_step_global_idx + 1

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
        reached_train_saving_step = (
            self.train_step_global_idx
            % trainer.model.save_train_predictions_every_n_step
            == 0
        )
        if trainer.model.should_save_preds and reached_train_saving_step:
            self.update_las_with_preds(outputs, "train")
            log.debug(
                f"Saving train preds to disk for batch number {self.train_step_global_idx}"
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

        if trainer.model.should_save_preds:
            if outputs is not None:
                self.update_las_with_preds(outputs, "val")

    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.model.train_iou_has_improved:
            log.debug(
                f"Saving validation preds to disk after train step {self.train_step_global_idx}.\n"
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
        if trainer.model.should_save_preds:
            log.debug(
                f"Saving test preds to disk after train step {self.train_step_global_idx}"
            )
            self.update_las_with_preds(outputs, "test")

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
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
                if self.in_memory_tile_filepath:
                    self.save_las_with_preds_and_close(phase)
                self.load_new_las_for_preds(filepath)
            else:
                with torch.no_grad():
                    self.assign_outputs_to_tile(batch, elem_idx, preds, proba, targets)

    def assign_outputs_to_tile(self, batch, elem_idx_list, preds, proba, targets):
        """Set the predicted elements in the current tile."""

        elem_points_idx = (batch.batch_y[..., None] == torch.Tensor(elem_idx_list)).any(
            -1
        )
        elem_pos = batch.pos_copy[elem_points_idx]
        elem_preds = preds[elem_points_idx]
        elem_proba = proba[elem_points_idx][:, 1]
        elem_targets = targets[elem_points_idx]

        assign_idx = knn(self.current_las_pos, elem_pos, k=1, num_workers=1)[1]

        self.current_las.BuildingsHasPredictions[assign_idx] = 1
        self.current_las.BuildingsPreds[assign_idx] = elem_preds
        self.current_las.BuildingsProba[assign_idx] = elem_proba.detach()
        elem_preds_confusion = self.get_confusion(elem_preds, elem_targets)
        self.current_las.BuildingsConfusion[assign_idx] = elem_preds_confusion

    def get_confusion(self, elem_preds, elem_targets):
        """Get a confusion vector: TN=0, Tp=1, FN=2, FP=3 - Nodata or Nan is 4."""
        elem_preds_confusion = elem_preds + 2 * elem_targets
        A = elem_preds * (elem_preds == elem_targets)
        B = (2 + elem_preds) * (elem_preds != elem_targets)
        elem_preds_confusion = A + B
        return elem_preds_confusion

    def load_new_las_for_preds(self, filepath):
        """Load a LAS and add necessary extradims."""

        self.current_las = laspy.read(filepath)
        self.in_memory_tile_filepath = filepath

        param = laspy.ExtraBytesParams(name="BuildingsPreds", type=int)
        self.current_las.add_extra_dim(param)
        self.current_las.BuildingsPreds[:] = 0

        param = laspy.ExtraBytesParams(name="BuildingsProba", type=float)
        self.current_las.add_extra_dim(param)
        self.current_las.BuildingsProba[:] = 0.0

        param = laspy.ExtraBytesParams(name="BuildingsConfusion", type=int)
        self.current_las.add_extra_dim(param)
        self.current_las.BuildingsConfusion[:] = 0

        param = laspy.ExtraBytesParams(name="BuildingsHasPredictions", type=int)
        self.current_las.add_extra_dim(param)
        self.current_las.BuildingsHasPredictions[:] = 0

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
        log.debug(f"Predictions save path is {output_path}.")
        # Closing:
        self.in_memory_tile_filepath = ""
        del self.current_las
        del self.current_las_pos
