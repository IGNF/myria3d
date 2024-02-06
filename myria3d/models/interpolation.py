import logging
import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pdal
import torch
from torch.distributions import Categorical
from torch_scatter import scatter_sum

from myria3d.pctl.dataset.utils import get_pdal_info_metadata, get_pdal_reader

log = logging.getLogger(__name__)


class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        interpolation_k: int = 10,
        classification_dict: Dict[int, str] = {},
        probas_to_save: Union[List[str], Literal["all"]] = "all",
        predicted_classification_channel: Optional[str] = "PredictedClassification",
        entropy_channel: Optional[str] = "entropy",
    ):
        """Initialization method.
        Args:
            interpolation_k (int, optional): Number of Nearest-Neighboors for inverse-distance averaging of logits. Defaults 10.
            classification_dict (Dict[int, str], optional): Mapper from classification code to class name (e.g. {6:building}). Defaults {}.
            probas_to_save (List[str] or "all", optional): Specific probabilities to save as new LAS dimensions.
            Override with None for no saving of probabilities. Defaults to "all".


        """

        self.k = interpolation_k
        self.classification_dict = classification_dict
        self.predicted_classification_channel = predicted_classification_channel
        self.entropy_channel = entropy_channel

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

        self.logits: List[torch.Tensor] = []
        self.idx_in_full_cloud_list: List[np.ndarray] = []

    def load_full_las_for_update(self, src_las: str, epsg: str) -> np.ndarray:
        """Loads a LAS and adds necessary extradim.

        Args:
            filepath (str): Path to LAS for which predictions are made.
            epsg (str): epsg to force the reading with
        """
        # We do not reset the dims we create channel.
        # Slight risk of interaction with previous values, but it is expected that all non-artefacts values are updated.

        pipeline = pdal.Pipeline() | get_pdal_reader(src_las, epsg)
        for proba_channel_to_create in self.probas_to_save:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{proba_channel_to_create}")
            pipeline |= pdal.Filter.assign(value=f"{proba_channel_to_create}=0")

        if self.predicted_classification_channel:
            # Copy from Classification to preserve data type
            # Also preserves values of artefacts.
            if self.predicted_classification_channel != "Classification":
                pipeline |= pdal.Filter.ferry(
                    dimensions=f"Classification=>{self.predicted_classification_channel}"
                )

        if self.entropy_channel:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{self.entropy_channel}")
            pipeline |= pdal.Filter.assign(value=f"{self.entropy_channel}=0")

        pipeline.execute()
        return pipeline.arrays[0]

    def store_predictions(self, logits, idx_in_original_cloud) -> None:
        """Keep a list of predictions made so far."""
        self.logits += [logits]
        self.idx_in_full_cloud_list += idx_in_original_cloud

    @torch.no_grad()
    def reduce_predicted_logits(self, nb_points) -> Tuple[torch.Tensor, np.ndarray]:
        """Interpolate logits to points without predictions using an inverse-distance weightning scheme.

        Returns:
            torch.Tensor, torch.Tensor: interpolated logits classification

        """

        # Concatenate elements from different batches
        logits: torch.Tensor = torch.cat(self.logits).cpu()
        idx_in_full_cloud: np.ndarray = np.concatenate(self.idx_in_full_cloud_list)
        del self.logits
        del self.idx_in_full_cloud_list

        # We scatter_sum logits based on idx, in case there are multiple predictions for a point.
        # scatter_sum reorders logits based on index,they therefore match las order.
        reduced_logits = torch.zeros((nb_points, logits.size(1)))
        scatter_sum(logits, torch.from_numpy(idx_in_full_cloud), out=reduced_logits, dim=0)
        # reduced_logits contains logits ordered by their idx in original cloud !
        # We need to select the points for which we have a prediction via idx_in_full_cloud.
        # NB1 : some points may not contain any predictions if they were in small areas.

        return reduced_logits[idx_in_full_cloud], idx_in_full_cloud

    @torch.no_grad()
    def reduce_predictions_and_save(self, raw_path: str, output_dir: str, epsg: str) -> str:
        """Interpolate all predicted probabilites to their original points in LAS file, and save.

        Args:
            interpolation (torch.Tensor, torch.Tensor): output of _interpolate, of which we need the logits.
            basename: str: file basename to save it with the same one
            output_dir (Optional[str], optional): Directory to save output LAS with new predicted classification, entropy,
            and probabilities. Defaults to None.
            epsg (str): epsg to force the reading with
        Returns:
            str: path of the updated, saved LAS file.

        """
        basename = os.path.basename(raw_path)
        # Read number of points only from las metadata in order to minimize memory usage
        nb_points = get_pdal_info_metadata(raw_path)["count"]
        logits, idx_in_full_cloud = self.reduce_predicted_logits(nb_points)

        probas = torch.nn.Softmax(dim=1)(logits)

        if self.predicted_classification_channel:
            preds = torch.argmax(logits, dim=1)
            preds = np.vectorize(self.reverse_mapper.get)(preds)

        del logits

        # Read las after fetching all information to write into it
        las = self.load_full_las_for_update(raw_path, epsg)

        for idx, class_name in enumerate(self.classification_dict.values()):
            if class_name in self.probas_to_save:
                # NB: Values for which we do not have a prediction (i.e. artefacts) get null probabilities.
                las[class_name][idx_in_full_cloud] = probas[:, idx]

        if self.predicted_classification_channel:
            # NB: Values for which we do not have a prediction (i.e. artefacts) keep their original class.
            las[self.predicted_classification_channel][idx_in_full_cloud] = preds
            log.info(
                f"Saving predicted classes to channel {self.predicted_classification_channel}."
                "Channel name can be changed by setting `predict.interpolator.predicted_classification_channel`."
            )
            del preds

        if self.entropy_channel:
            # NB: Values for which we do not have a prediction (i.e. artefacts) get null entropy.
            las[self.entropy_channel][idx_in_full_cloud] = Categorical(probs=probas).entropy()
            log.info(
                f"Saving Shannon entropy of probabilities to channel {self.entropy_channel}."
                "Channel name can be changed by setting `predict.interpolator.entropy_channel`"
            )
        del idx_in_full_cloud

        os.makedirs(output_dir, exist_ok=True)
        out_f = os.path.join(output_dir, basename)
        out_f = os.path.abspath(out_f)
        log.info(f"Updated LAS ({basename}) will be saved to: \n {output_dir}\n")
        log.info("Saving...")
        pipeline = pdal.Writer.las(
            filename=out_f, extra_dims="all", minor_version=4, dataformat_id=8
        ).pipeline(las)
        pipeline.execute()
        log.info("Saved.")

        return out_f
