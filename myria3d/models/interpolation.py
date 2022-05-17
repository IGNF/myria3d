"""How we turn from prediction made on a subsampled subset of a Las to a complete point cloud."""

import os
from typing import Dict, List, Optional, Literal, Union

import pdal
import numpy as np
import torch
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from myria3d.utils import utils
from torch.distributions import Categorical

from myria3d.data.transforms import ChannelNames

log = utils.get_logger(__name__)


class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        interpolation_k: int = 10,
        classification_dict: Dict[int, str] = {},
        probas_to_save: Union[List[str], Literal["all"]] = "all",
        output_dir: Optional[str] = None,
    ):
        """Initialization method.

        Args:
            interpolation_k (int, optional): Number of Nearest-Neighboors for inverse-distance averaging of logits. Defaults 10.
            classification_dict (Dict[int, str], optional): Mapper from classification code to class name (e.g. {6:building}). Defaults {}.
            probas_to_save (List[str] or "all", optional): Specific probabilities to save as new LAS dimensions.
            Override with None for no saving of probabilitiues. Defaults to "all".
            output_dir (Optional[str], optional): Directory to save output LAS with new predicted classification, entropy,
            and probabilities. Defaults to None.

        """
        self.output_dir = output_dir
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        self.k = interpolation_k
        self.classification_dict = classification_dict

        if probas_to_save == "all":
            self.probas_to_save = list(classification_dict.values())
        elif probas_to_save is None:
            self.probas_to_save = []
        else:
            self.probas_to_save = probas_to_save

        # Maps ascending index (0,1,2,...) back to conventionnal LAS classification codes (6=buildings, etc.)
        self.reverse_mapper: Dict[int, int] = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }

        # Tracker for current processed file.
        self.current_f = ""

    def _load_las(self, filepath: str):
        """Loads a LAS and adds necessary extradim.

        Args:
            filepath (str): Path to LAS for which predictions are made.

        """
        self.current_f = filepath
        pipeline = pdal.Reader.las(filename=filepath)

        new_dims = self.probas_to_save + [
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
        """Keep a list of predictions made so far.
        In Test phase, interpolation and saving are trigerred when a new file is encountered.

        Args:
            outputs (dict): Outputs of lightning's predict_step or test_step.

        Returns:
            List[interpolation]: list of interpolation made for these specific outputs. This is typically empty,
            except when we switch from a LAS to another, at which point we need to output the result of the interpolation
            for IoU logging by a callback.

        """
        _itps = []

        batch = outputs["batch"].detach()
        logits_b = outputs["logits"].detach()
        some_targets_to_interpolate = "y_copy" in batch
        if some_targets_to_interpolate:
            # TODO: it seems that this is always done due to data transforms.
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
        """Interpolate logits to points without predictions using an inverse-distance weightning scheme.

        Returns:
            torch.Tensor, torch.Tensor: interpolated logits and targets/original classification

        """

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
    def _write(self, interpolation) -> str:
        """Interpolate all predicted probabilites to their original points in LAS file, and save.

        Args:
            interpolation (torch.Tensor, torch.Tensor): output of _interpolate, of which we need the logits.

        Returns:
            str: path of the updated, saved LAS file.

        """

        basename = os.path.basename(self.current_f)
        out_f = os.path.join(self.output_dir, basename)
        log.info(f"Updated LAS will be saved to {out_f}")

        logits, _ = interpolation

        probas = torch.nn.Softmax(dim=1)(logits)
        for idx, class_name in enumerate(self.classification_dict.values()):
            if class_name in self.probas_to_save:
                self.las[class_name][:] = probas[:, idx]

        preds = torch.argmax(logits, dim=1)
        preds = np.vectorize(self.reverse_mapper.get)(preds)
        self.las[ChannelNames.PredictedClassification.value][:] = preds

        self.las[ChannelNames.ProbasEntropy.value][:] = Categorical(
            probs=probas
        ).entropy()

        log.info("Saving...")

        pipeline = pdal.Writer.las(
            filename=out_f, extra_dims="all", minor_version=4, dataformat_id=8
        ).pipeline(self.las)
        pipeline.execute()
        log.info("Saved.")

        return out_f

    def interpolate_and_save(self):
        """Interpolate and save in a single method, for predictions."""
        interpolation = self._interpolate()
        out_f = self._write(interpolation)

        return out_f
