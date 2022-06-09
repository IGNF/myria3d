from enum import Enum
import logging
import os
from typing import Dict, List, Literal, Union
import pdal
import numpy as np
import torch
from torch_geometric.nn.unpool import knn_interpolate
from torch.distributions import Categorical
from torch_scatter import scatter_sum

log = logging.getLogger(__name__)


class ChannelNames(Enum):
    """Names of custom additional LAS channel."""

    PredictedClassification = "PredictedClassification"
    ProbasEntropy = "entropy"


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
            Override with None for no saving of probabilitiues. Defaults to "all".


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
        self.reverse_mapper: Dict[int, int] = {
            class_index: class_code
            for class_index, class_code in enumerate(classification_dict.keys())
        }
        self.logits_sub: List[torch.Tensor] = []
        self.pos_sub: List[torch.Tensor] = []
        self.batch_sub: List[torch.Tensor] = []
        self.idx_in_full_cloud_list: List[np.ndarray] = []

    def load_full_las_for_update(self, raw_path: str):
        """Loads a LAS and adds necessary extradim.

        Args:
            filepath (str): Path to LAS for which predictions are made.
        """
        # self.current_f = filepath
        pipeline = pdal.Reader.las(filename=raw_path)
        new_dims = self.probas_to_save + [
            ChannelNames.PredictedClassification.value,
            ChannelNames.ProbasEntropy.value,
        ]
        for new_dim in new_dims:
            pipeline |= pdal.Filter.ferry(
                dimensions=f"=>{new_dim}"
            ) | pdal.Filter.assign(value=f"{new_dim}=0")
        pipeline.execute()
        return pipeline.arrays[0]  # named array

    @torch.no_grad()
    def store_predictions(self, logits, pos, batch, idx_in_original_cloud):
        """Keep a list of predictions made so far."""
        self.logits_sub.append(logits)
        self.pos_sub.append(pos)
        self.idx_in_full_cloud_list += idx_in_original_cloud
        if not self.batch_sub:
            # starts at 0 if this is the first batch
            self.batch_sub.append(batch)
        else:
            # starts from current max batch index
            current_max_batch_idx = max(max(b) for b in self.batch_sub)
            self.batch_sub.append(current_max_batch_idx + 1 + batch)

    @torch.no_grad()
    def interpolate_logits(self, las):
        """Interpolate logits to points without predictions using an inverse-distance weightning scheme.

        Returns:
            torch.Tensor, torch.Tensor: interpolated logits and targets/original classification

        """

        # Cat
        logits_sub: torch.Tensor = torch.cat(self.logits_sub).cpu()
        pos_sub: torch.Tensor = torch.cat(self.pos_sub).cpu()
        batch_sub: torch.Tensor = torch.cat(self.batch_sub).cpu()
        del self.logits_sub
        del self.batch_sub

        # create a batch for the full las
        # concatenate
        batch_full_cloud: torch.Tensor = torch.cat(
            [
                torch.full((len(a),), i)
                for i, a in enumerate(self.idx_in_full_cloud_list)
            ]
        )
        idx_in_full_cloud: np.ndarray = np.concatenate(self.idx_in_full_cloud_list)
        del self.idx_in_full_cloud_list

        # Reorganize points in original LAS to order them by batch,
        # matching subsampled batch
        # If required, they could be reorered back at save time by using
        # np.argsort(current_max_batch_idx) as sorting indices.

        ordered_las = las[idx_in_full_cloud]

        pos_las = torch.from_numpy(
            np.asarray(
                [
                    ordered_las["X"],
                    ordered_las["Y"],
                    ordered_las["Z"],
                ],
                dtype=np.float32,
            ).transpose()
        )

        # Find nn among points with predictions for all points
        # Only interpolate within a model's receptive field zone
        logits = knn_interpolate(
            logits_sub,
            pos_sub,
            pos_las,
            batch_x=batch_sub,
            batch_y=batch_full_cloud,
            k=self.k,
            num_workers=4,
        )

        # We scatter_sum logits based on idx, in case there are multiple predictions for a point.
        # scatter_sum reorders logitsbased on index,they therefore match las order.
        logits = scatter_sum(logits, torch.from_numpy(idx_in_full_cloud), dim=0)
        idx_in_full_las = np.sort(np.unique(idx_in_full_cloud))
        return logits, idx_in_full_las

    @torch.no_grad()
    def write(self, raw_path: str, output_dir: str) -> str:
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
        las = self.load_full_las_for_update(raw_path=raw_path)
        logits, idx_in_full_las = self.interpolate_logits(las)

        probas = torch.nn.Softmax(dim=1)(logits)
        for idx, class_name in enumerate(self.classification_dict.values()):
            if class_name in self.probas_to_save:
                las[class_name][idx_in_full_las] = probas[:, idx]

        preds = torch.argmax(logits, dim=1)
        preds = np.vectorize(self.reverse_mapper.get)(preds)
        las[ChannelNames.PredictedClassification.value][idx_in_full_las] = preds

        las[ChannelNames.ProbasEntropy.value][idx_in_full_las] = Categorical(
            probs=probas
        ).entropy()

        os.makedirs(output_dir, exist_ok=True)
        out_f = os.path.join(output_dir, basename)
        out_f = os.path.abspath(out_f)
        log.info(f"Updated LAS will be saved to {out_f}.")
        log.info("Saving...")
        pipeline = pdal.Writer.las(
            filename=out_f, extra_dims="all", minor_version=4, dataformat_id=8
        ).pipeline(las)
        pipeline.execute()
        log.info("Saved.")

        return out_f
