import os
from typing import Dict, List

import laspy
import numpy as np
import torch
from torch_geometric.nn.pool import knn

from lidar_multiclass.utils import utils
from lidar_multiclass.utils import utils

from lidar_multiclass.datamodules.transforms import ChannelNames

log = utils.get_logger(__name__)


class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        output_dir: str,
        classification_dict: Dict[int, str],
        names_of_probas_to_save: List[str] = [],
    ):

        os.makedirs(output_dir, exist_ok=True)
        self.preds_dirpath = output_dir
        self.current_las_filepath = ""
        self.classification_dict = classification_dict
        self.names_of_probas_to_save = names_of_probas_to_save

        self.reverse_classification_mapper = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }

        self.index_of_probas_to_save = [
            list(classification_dict.values()).index(name)
            for name in names_of_probas_to_save
        ]

    @torch.no_grad()
    # TODO: interpolate_and_save to return a path that is appended to a list.
    # TODO: only one file, systematically -> OK for predict, and elsewise dealt with in on_test_batch_end of callback.
    def update_with_inference_outputs(self, outputs: dict):
        """
        Save the predicted classes in las format with position.
        Handle las loading when necessary.

        :param outputs: outputs of a step.
        """
        batch = outputs["batch"].detach()
        batch_classification = outputs["preds"].detach()
        if "entropy" in outputs:
            batch_entropy = outputs["entropy"].detach()
        batch_probas = outputs["proba"][:, self.index_of_probas_to_save].detach()
        for batch_idx, las_filepath in enumerate(batch.las_filepath):
            is_a_new_tile = las_filepath != self.current_las_filepath
            if is_a_new_tile:
                close_previous_las_first = self.current_las_filepath != ""
                if close_previous_las_first:
                    self.interpolate_and_save()
                self._load_las_for_classification_update(las_filepath)

            idx_x = batch.batch_x == batch_idx
            self.updates_classification_subsampled.append(batch_classification[idx_x])
            self.updates_probas_subsampled.append(batch_probas[idx_x])
            if "entropy" in outputs:
                self.updates_entropy_subsampled.append(batch_entropy[idx_x])
            self.updates_pos_subsampled.append(batch.pos_copy_subsampled[idx_x])
            idx_y = batch.batch_y == batch_idx
            self.updates_pos.append(batch.pos_copy[idx_y])

    # TODO: make sure that called only for prediction ?
    def _load_las_for_classification_update(self, filepath):
        """Load a LAS and add necessary extradim."""

        self.las = laspy.read(filepath)
        self.current_las_filepath = filepath

        coln = ChannelNames.PredictedClassification.value
        param = laspy.ExtraBytesParams(name=coln, type=int)
        self.las.add_extra_dim(param)
        self.las[coln][:] = 0

        param = laspy.ExtraBytesParams(
            name=ChannelNames.ProbasEntropy.value, type=float
        )
        self.las.add_extra_dim(param)
        self.las[ChannelNames.ProbasEntropy.value][:] = 0.0

        for class_name in self.names_of_probas_to_save:
            param = laspy.ExtraBytesParams(name=class_name, type=float)
            self.las.add_extra_dim(param)
            self.las[class_name][:] = 0.0

        self.las_pos = torch.from_numpy(
            np.asarray(
                [
                    self.las.x,
                    self.las.y,
                    self.las.z,
                ],
                dtype=np.float32,
            ).transpose()
        )
        # TODO: move these declaration to init
        self.updates_classification_subsampled = []
        self.updates_probas_subsampled = []
        self.updates_entropy_subsampled = []
        self.updates_pos_subsampled = []
        self.updates_pos = []

    # TODO: Reduce values to have unique by pos with torch_scatter -> find mean to index pos with integers
    # -> simply sort them ?
    # TODO: separate interpolate from save (two steps), and remapping happens before saving ?

    @torch.no_grad()
    def interpolate_and_save(self):
        """
        Interpolate all predicted probabilites to their original points in LAS file, and save.
        Returns the path of the updated, saved LAS file.
        """

        basename = os.path.basename(self.current_las_filepath)

        os.makedirs(self.preds_dirpath, exist_ok=True)
        self.output_path = os.path.join(self.preds_dirpath, basename)
        log.info(f"Updated LAS will be saved to {self.output_path}")

        # Cat
        self.updates_pos = torch.cat(self.updates_pos).cpu()
        self.updates_pos_subsampled = torch.cat(self.updates_pos_subsampled).cpu()
        self.updates_probas_subsampled = torch.cat(self.updates_probas_subsampled).cpu()
        self.updates_classification_subsampled = torch.cat(
            self.updates_classification_subsampled
        ).cpu()
        if len(self.updates_entropy_subsampled):
            self.updates_entropy_subsampled = torch.cat(
                self.updates_entropy_subsampled
            ).cpu()

        # Remap predictions to good classification codes
        self.updates_classification_subsampled = np.vectorize(
            self.reverse_classification_mapper.get
        )(self.updates_classification_subsampled)
        self.updates_classification_subsampled = torch.from_numpy(
            self.updates_classification_subsampled
        )

        # Find nn among points with predictions for all points
        assign_idx = knn(
            self.updates_pos_subsampled,
            self.las_pos,
            k=1,
            num_workers=1,
        )[1]

        # Interpolate predictions
        self.updates_classification = self.updates_classification_subsampled[assign_idx]
        self.updates_probas = self.updates_probas_subsampled[assign_idx]
        if len(self.updates_entropy_subsampled):
            self.updates_entropy = self.updates_entropy_subsampled[assign_idx]

        # Only update channels for points with a predicted point that is close enough
        nn_pos = self.updates_pos_subsampled[assign_idx]
        euclidian_distance = torch.sqrt(((self.las_pos - nn_pos) ** 2).sum(axis=1))
        INTERPOLATION_RADIUS = 2.5
        close_enough_with_preds = euclidian_distance < INTERPOLATION_RADIUS

        for class_idx_in_tensor, class_name in enumerate(self.names_of_probas_to_save):
            self.las[class_name][close_enough_with_preds] = self.updates_probas[
                close_enough_with_preds, class_idx_in_tensor
            ]
        self.las[ChannelNames.PredictedClassification.value][
            close_enough_with_preds
        ] = self.updates_classification[close_enough_with_preds]
        if len(self.updates_entropy):
            self.las[ChannelNames.ProbasEntropy.value][
                close_enough_with_preds
            ] = self.updates_entropy[close_enough_with_preds]

        log.info(f"Saving...")
        self.las.write(self.output_path)
        log.info(f"Saved.")

        # Clean-up - get rid of current data to go easy on memory
        self.current_las_filepath = ""
        del self.las
        del self.las_pos
        del self.updates_pos_subsampled
        del self.updates_pos
        del self.updates_classification_subsampled
        del self.updates_classification
        del self.updates_probas_subsampled

        return self.output_path
