from enum import Enum
import logging
import os
from typing import Dict, List, Literal, Union
import pdal
import numpy as np
import torch
from torch.distributions import Categorical
from torch_scatter import scatter_sum

from myria3d.pctl.dataset.utils import get_pdal_reader

log = logging.getLogger(__name__)


class ChannelNames(Enum):
    """Names of custom additional LAS channel."""

    PredictedClassification = "PredictedClassification"
    ProbasEntropy = "entropy"


@torch.no_grad()
class Interpolator:
    """A class to load, update with classification, update with probas (optionnal), and save a LAS."""

    def __init__(
        self,
        interpolation_k: int = 10,
        classification_dict: Dict[int, str] = {},
        probas_to_save: Union[List[str], Literal["all"]] = "all",
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

        if probas_to_save == "all":
            self.probas_to_save = list(classification_dict.values())
        elif probas_to_save is None:
            self.probas_to_save = []
        else:
            self.probas_to_save = probas_to_save

        # Maps ascending index (0,1,2,...) back to conventionnal LAS classification codes (6=buildings, etc.)
        self.reverse_mapper: Dict[int, int] = {class_index: class_code for class_index, class_code in enumerate(classification_dict.keys())}

        self.logits: List[torch.Tensor] = []
        self.idx_in_full_cloud_list: List[np.ndarray] = []

    def load_full_las_for_update(self, src_las: str):
        """Loads a LAS and adds necessary extradim.

        Args:
            filepath (str): Path to LAS for which predictions are made.
        """
        # self.current_f = filepath
        pipeline = get_pdal_reader(src_las)
        new_dims = self.probas_to_save + [
            ChannelNames.PredictedClassification.value,
            ChannelNames.ProbasEntropy.value,
        ]
        for new_dim in new_dims:
            pipeline |= pdal.Filter.ferry(dimensions=f"=>{new_dim}") | pdal.Filter.assign(value=f"{new_dim}=0")
        pipeline.execute()
        return pipeline.arrays[0]  # named array

    def store_predictions(self, logits, idx_in_original_cloud):
        """Keep a list of predictions made so far."""
        self.logits += [logits]
        self.idx_in_full_cloud_list += idx_in_original_cloud

    @torch.no_grad()
    def reduce_predicted_logits(self, las):
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
        # scatter_sum reorders logitsbased on index,they therefore match las order.
        reduced_logits = torch.zeros((len(las), logits.size(1)))
        scatter_sum(logits, torch.from_numpy(idx_in_full_cloud), out=reduced_logits, dim=0)
        # reduced_logits contains logits ordered by their idx in original cloud !
        # Warning : some points may not contain any predictions if they were in small areas.
        return reduced_logits

    @torch.no_grad()
    def reduce_predictions_and_save(self, raw_path: str, output_dir: str) -> str:
        """Interpolate all predicted probabilites to their original points in LAS file, and save.

        Args:
            interpolation (torch.Tensor, torch.Tensor): output of _interpolate, of which we need the logits.
            basename: str: file basename to save it with the same one
            output_dir (Optional[str], optional): Directory to save output LAS with new predicted classification, entropy,
            and probabilities. Defaults to None.
        Returns:
            str: path of the updated, saved LAS file.

        """
        basename = os.path.basename(raw_path)
        las = self.load_full_las_for_update(src_las=raw_path)
        logits = self.reduce_predicted_logits(las)

        probas = torch.nn.Softmax(dim=1)(logits)
        for idx, class_name in enumerate(self.classification_dict.values()):
            if class_name in self.probas_to_save:
                las[class_name] = probas[:, idx]

        preds = torch.argmax(logits, dim=1)
        preds = np.vectorize(self.reverse_mapper.get)(preds)
        las[ChannelNames.PredictedClassification.value] = preds

        las[ChannelNames.ProbasEntropy.value] = Categorical(probs=probas).entropy()

        os.makedirs(output_dir, exist_ok=True)
        out_f = os.path.join(output_dir, basename)
        out_f = os.path.abspath(out_f)
        log.info(f"Updated LAS ({basename}) will be saved to \n {output_dir}\n")
        log.info("Saving...")
        pipeline = pdal.Writer.las(filename=out_f, extra_dims="all", minor_version=4, dataformat_id=8).pipeline(las)
        pipeline.execute()
        log.info("Saved.")

        return out_f
