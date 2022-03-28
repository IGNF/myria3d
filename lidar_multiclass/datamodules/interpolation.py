import os
from tokenize import Number
from typing import Dict, List, Optional

import pdal
import numpy as np
import torch
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from lidar_multiclass.utils import utils
from lidar_multiclass.utils import utils
from torch.distributions import Categorical

from lidar_multiclass.datamodules.transforms import ChannelNames

log = utils.get_logger(__name__)


class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        classification_dict: Dict[int, str] = {},
        interpolation_k: Number = 10,
        output_dir: Optional[str] = None,
        names_of_probas_to_save: List[str] = [],
    ):
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.classification_dict = classification_dict
        self.probas_names = names_of_probas_to_save
        self.current_f = ""
        self.k = interpolation_k

        self.reverse_mapper = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }

        self.probas_idx = [
            list(classification_dict.values()).index(name)
            for name in names_of_probas_to_save
        ]

    def _load_las(self, filepath: str):
        """Load a LAS and add necessary extradim."""
        self.current_f = filepath
        pipeline = pdal.Reader.las(filename=filepath)

        new_dims = self.probas_names + [
            ChannelNames.PredictedClassification.value,
            ChannelNames.ProbasEntropy.value,
        ]
        for new_dim in new_dims:
            pipeline |= pdal.Filter.ferry(
                dimensions=f"=>{new_dim}"
            ) | pdal.Filter.assign(value=f"{new_dim}=0")
        pipeline.execute()
        self.las = pipeline.arrays[0]  # named array

        self.pos_las = torch.from_numpy(
            np.asarray(
                [
                    self.las["X"],
                    self.las["Y"],
                    self.las["Z"],
                ],
                dtype=np.float32,
            ).transpose()
        )
        self.logits_sub_l = []
        self.targets_l = []
        self.pos_sub_l = []
        self.pos_l = []

    @torch.no_grad()
    def update(self, outputs: dict):
        """
        Save the predicted classes in las format with position.
        Handle las loading when necessary.

        :param outputs: outputs of a step.
        returns:
          list[str]: when
        """
        _itps = []

        batch = outputs["batch"].detach()
        logits_b = outputs["logits"].detach()
        some_targets_to_interpolate = "y_copy" in batch
        if some_targets_to_interpolate:
            targets_b = batch.y_copy.detach()

        for batch_idx, las_filepath in enumerate(batch.las_filepath):
            is_a_new_tile = las_filepath != self.current_f
            if is_a_new_tile:
                close_previous_las_first = self.current_f != ""
                if close_previous_las_first:
                    interpolation = self._interpolate()
                    if self.output_dir:
                        self._write(interpolation)
                    _itps += [interpolation]
                self._load_las(las_filepath)

            # subsampled elements
            idx_x = batch.batch_x == batch_idx
            self.logits_sub_l.append(logits_b[idx_x])
            self.pos_sub_l.append(batch.pos_copy_subsampled[idx_x])

            if some_targets_to_interpolate:
                # non-sampled elements
                idx_y = batch.batch_y == batch_idx
                self.pos_l.append(batch.pos_copy[idx_y])
                self.targets_l.append(targets_b[idx_y])

        return _itps

    def _interpolate(self):
        # Cat
        pos_sub = torch.cat(self.pos_sub_l).cpu()
        logits_sub = torch.cat(self.logits_sub_l).cpu()

        # Find nn among points with predictions for all points
        logits = knn_interpolate(
            logits_sub,
            pos_sub,
            self.pos_las,
            batch_x=None,
            batch_y=None,
            k=self.k,
            num_workers=4,
        )
        # If no target, returns interpolared logits (i.e. at predict time)
        if not self.targets_l:
            return logits, None

        # Interpolate non-sampled targets if present (i.e. at test time)
        targets = torch.cat(self.targets_l).cpu()
        pos = torch.cat(self.pos_l).cpu()
        assign_idx = knn(pos, self.pos_las, k=1, num_workers=4)
        _, x_idx = assign_idx
        targets = targets[x_idx]

        return logits, targets

    @torch.no_grad()
    def _write(self, interpolation):
        """
        Interpolate all predicted probabilites to their original points in LAS file, and save.
        Returns the path of the updated, saved LAS file.
        """

        basename = os.path.basename(self.current_f)
        out_f = os.path.join(self.output_dir, basename)
        log.info(f"Updated LAS will be saved to {out_f}")

        logits, _ = interpolation

        probas = torch.nn.Softmax(dim=1)(logits)
        for idx, class_name in enumerate(self.probas_names):
            self.las[class_name][:] = probas[:, idx]

        preds = torch.argmax(logits, dim=1)
        preds = np.vectorize(self.reverse_mapper.get)(preds)
        self.las[ChannelNames.PredictedClassification.value][:] = preds

        entropy = Categorical(probs=probas).entropy()
        self.las[ChannelNames.ProbasEntropy.value][:] = entropy

        log.info(f"Saving...")

        pipeline = pdal.Writer.las(
            filename=out_f, extra_dims=f"all", minor_version=4, dataformat_id=8
        ).pipeline(self.las)
        pipeline.execute()
        log.info(f"Saved.")

        return out_f

    def interpolate_and_save(self):
        """Interpolate and save in a single method, for predictions."""
        interpolation = self._interpolate()
        out_f = self._write(interpolation)

        return out_f
