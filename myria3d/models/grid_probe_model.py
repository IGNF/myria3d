"""Linear probing on a frozen Flair3D+ multitask backbone.

Trains N independent linear heads (`probe_lrs`) sharing one frozen-backbone forward
pass per step and one `manual_backward()` over their summed losses -- heads have
disjoint parameters, so this gives every head its own correct gradient from a single
pass, exactly like Pointcept's `GridProbeSegmentorV2` / `tools/grid_then_seeds.py`.
Used for both phases of the sweep-LR-then-10-seeds protocol: a 12-value LR grid
sweep (`GridProbeWinnerSelector` picks the winner) and, at the winning LR, 10
independently-initialized seed heads (`GridProbeSeedEnsembleTester` aggregates test
metrics) -- see `myria3d/callbacks/grid_probe_callbacks.py` and
readme_linear_probing.md.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch_geometric.data import Batch

from myria3d.models.modules.hypercolumn import (
    DEFAULT_SCALES,
    STAGE_CHANNELS,
    hypercolumn_concat,
)
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask
from myria3d.pctl.transforms.transforms import (
    COLOR_FEATURE_NAMES,
    STRENGTH_FEATURE_NAMES,
    resolve_x_feature_names,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)

# Matches PyGRandLANetMultiTask.backbone_parameters()'s module list -- everything
# needed by _forward_encoder_stages, nothing from any task head.
BACKBONE_MODULE_NAMES = (
    "fc0",
    "block1",
    "block2",
    "block3",
    "block4",
    "mlp_summit",
    "fp4",
    "fp3",
    "fp2",
    "fp1",
)
# Learned RGB / intensity fill-in values (see multitask_model.py::_init_learned_masked_feat)
# are attributes of the LightningModule that trained the checkpoint, not of the
# backbone nn.Module -- hence no "model." prefix in the checkpoint state_dict.
LEARNED_MASK_PARAM_NAMES = ("color_mask_value", "strength_mask_value")
_PLACEHOLDER_TASK_CONFIGS = {"_probe_placeholder": {"task_type": "semantic", "num_classes": 2}}


def load_frozen_backbone(
    ckpt_path: str,
    num_features: int = 5,
    num_neighbors: int = 16,
    decimation: int = 4,
) -> Tuple[PyGRandLANetMultiTask, Dict[str, Tensor]]:
    """Load a Flair3D+ multitask checkpoint's frozen backbone + learned mask values.

    The checkpoint's task heads are irrelevant to linear probing (only
    ``_forward_encoder_stages`` is ever called) so ``PyGRandLANetMultiTask`` is built
    with a throwaway single-task config -- its head shapes don't matter, only its
    backbone shapes (``fc0``/``fp1``'s ``d_bottleneck``) do, and those only depend on
    ``num_features`` and the max num_classes across pretraining tasks being <= 32
    (true for Flair3D+'s 16-class segment task), so this always reconstructs the
    same backbone shapes as the real pretraining run.

    Returns the backbone module (eval-mode, ``requires_grad=False`` on every
    parameter) and a ``{color_mask_value, strength_mask_value}`` dict (empty if the
    pretrained run had ``learned_masked_feat: false``).
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    backbone = PyGRandLANetMultiTask(
        num_features=num_features,
        task_configs=_PLACEHOLDER_TASK_CONFIGS,
        decimation=decimation,
        num_neighbors=num_neighbors,
    )
    backbone_state_dict = {
        key[len("model.") :]: value
        for key, value in state_dict.items()
        if key.startswith("model.") and key.split(".")[1] in BACKBONE_MODULE_NAMES
    }
    if not backbone_state_dict:
        raise ValueError(
            f"No backbone weights (keys prefixed 'model.{{{','.join(BACKBONE_MODULE_NAMES)}}}') "
            f"found in checkpoint {ckpt_path}. Is this a Flair3D+ multitask checkpoint?"
        )
    backbone.load_state_dict(backbone_state_dict, strict=False)

    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()

    mask_values = {
        name: state_dict[name].clone() for name in LEARNED_MASK_PARAM_NAMES if name in state_dict
    }
    return backbone, mask_values


def apply_learned_mask_fill(batch: Batch, mask_values: Dict[str, Tensor]) -> Tensor:
    """Replace fully-absent-modality channels with the frozen backbone's own learned
    fill-in value instead of a naive zero. ``mask_values`` is the
    ``{color_mask_value, strength_mask_value}`` dict returned by
    ``load_frozen_backbone`` (only present when the pretrained run had
    ``learned_masked_feat: true``); shared by ``GridProbeModel`` and
    ``scripts/extract_pureforest_pooled_features.py`` so the fill logic isn't
    duplicated between the LightningModule and the standalone script.
    """
    x = batch.x
    if not mask_values:
        return x
    names = resolve_x_feature_names(batch)
    if not names or len(names) != x.size(1):
        return x
    out = x
    for feat_names, mask_key, param_name in (
        (COLOR_FEATURE_NAMES, "color_mask", "color_mask_value"),
        (STRENGTH_FEATURE_NAMES, "strength_mask", "strength_mask_value"),
    ):
        if param_name not in mask_values:
            continue
        mask = getattr(batch, mask_key, None)
        if mask is None:
            continue
        present = [name for name in feat_names if name in names]
        if not present:
            continue
        param = mask_values[param_name].to(dtype=out.dtype, device=out.device)
        point_mask = mask.bool().reshape(-1).to(device=out.device)
        cols = []
        for i, name in enumerate(names):
            col = out[:, i]
            if name in present:
                fill = param[:, feat_names.index(name)].reshape(())
                col = torch.where(point_mask, fill, col)
            cols.append(col)
        out = torch.stack(cols, dim=1)
    return out


class ProbeHead(nn.Module):
    """Optional norm -> optional dropout -> single Linear. Mirrors Pointcept's
    `ProbeHead`; only the LR axis is exercised for DALES/H3D/ECLAIR's 12-probe sweep,
    so `norm`/`dropout` are wired but left at their no-op defaults (extension point,
    not built out further)."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        norm: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        if norm == "batch_norm":
            layers.append(nn.BatchNorm1d(in_dim))
        elif norm == "layer_norm":
            layers.append(nn.LayerNorm(in_dim))
        elif norm is not None:
            raise ValueError(
                f"Unknown ProbeHead norm {norm!r}, expected None/batch_norm/layer_norm"
            )
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GridProbeModel(LightningModule):
    """LightningModule training N linear probes on a frozen backbone's multi-scale
    encoder hypercolumn feature. See module docstring for the two-phase protocol."""

    def __init__(
        self,
        backbone_ckpt_path: str,
        probe_lrs: Sequence[float],
        num_classes: int,
        ignore_index: Optional[int] = None,
        scale_blocks: Sequence[str] = DEFAULT_SCALES,
        num_features: int = 5,
        num_neighbors: int = 16,
        decimation: int = 4,
        interpolation_k: int = 1,
        probe_norm: Optional[str] = None,
        probe_dropout: float = 0.0,
        optimizer: Optional[functools.partial] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.backbone, mask_values = load_frozen_backbone(
            backbone_ckpt_path,
            num_features=num_features,
            num_neighbors=num_neighbors,
            decimation=decimation,
        )
        for name, value in mask_values.items():
            self.register_parameter(name, nn.Parameter(value, requires_grad=False))
        self.learned_masked_feat = bool(mask_values)

        self.scale_blocks: Tuple[str, ...] = tuple(scale_blocks)
        self.hypercolumn_dim = sum(STAGE_CHANNELS[s] for s in self.scale_blocks)

        self.optimizer_factory = optimizer or functools.partial(torch.optim.Adam)

        probe_lrs = list(probe_lrs)
        if not probe_lrs:
            raise ValueError("probe_lrs must contain at least one learning rate.")
        # ModuleDict keys can't contain "." -- scientific notation (`.0e`) never does.
        self.probe_names: List[str] = [f"probe{i}_lr{lr:.0e}" for i, lr in enumerate(probe_lrs)]
        self.probe_lrs: Dict[str, float] = dict(zip(self.probe_names, probe_lrs))
        self.probes = nn.ModuleDict(
            {
                name: ProbeHead(
                    self.hypercolumn_dim, num_classes, norm=probe_norm, dropout=probe_dropout
                )
                for name in self.probe_names
            }
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index if ignore_index is not None else -100
        )
        # None = all probes active (grid sweep / seed ensemble training); a single
        # probe name = only that probe is evaluated (set by GridProbeWinnerSelector
        # after the grid phase, so the automatic post-fit `trainer.test()` in
        # train.py only scores the winner).
        self.active_probe: Optional[str] = None

    def train(self, mode: bool = True) -> "GridProbeModel":
        super().train(mode)
        # Backbone BatchNorm must stay pinned to eval-mode running stats even though
        # Lightning calls `.train()` on the whole module every epoch (Pointcept's
        # `bn_eval_mode` override) -- the backbone is frozen, its BN stats should
        # never drift, and small probe-training batches would otherwise corrupt them.
        self.backbone.eval()
        return self

    def _active_probe_names(self) -> List[str]:
        return [self.active_probe] if self.active_probe is not None else self.probe_names

    def _fill_masked_features(self, batch: Batch) -> Tensor:
        """Replace fully-absent-modality channels (DALES: color, H3D: strength) with
        the frozen backbone's own learned fill-in value instead of a naive zero --
        see DownstreamNpyDataset / readme_linear_probing.md."""
        if not self.learned_masked_feat:
            return batch.x
        mask_values = {
            name: getattr(self, name) for name in LEARNED_MASK_PARAM_NAMES if hasattr(self, name)
        }
        return apply_learned_mask_fill(batch, mask_values)

    def _hypercolumn_features(self, batch: Batch) -> Tensor:
        x = self._fill_masked_features(batch)
        with torch.no_grad():
            stages = self.backbone._forward_encoder_stages(x, batch.pos, batch.batch, batch.ptr)
            feat = hypercolumn_concat(
                stages,
                batch.pos,
                batch.batch,
                scales=self.scale_blocks,
                k=self.hparams.interpolation_k,
            )
        return feat

    def _probe_logits(self, feat: Tensor, probe_names: List[str]) -> Dict[str, Tensor]:
        return {name: self.probes[name](feat) for name in probe_names}

    def _probe_losses(self, logits: Dict[str, Tensor], targets: Tensor) -> Dict[str, Tensor]:
        targets = targets.long()
        return {
            name: self.criterion(probe_logits, targets) for name, probe_logits in logits.items()
        }

    def training_step(self, batch: Batch, batch_idx: int) -> dict:
        feat = self._hypercolumn_features(batch)
        probe_names = self._active_probe_names()
        logits = self._probe_logits(feat, probe_names)
        losses = self._probe_losses(logits, batch.y)
        total_loss = sum(losses.values())

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        opt_by_name = dict(zip(self.probe_names, optimizers))
        active_opts = [opt_by_name[name] for name in probe_names]

        for opt in active_opts:
            opt.zero_grad()
        self.manual_backward(total_loss)
        for opt in active_opts:
            opt.step()

        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=False)
        for name, loss in losses.items():
            self.log(f"train/loss_{name}", loss, on_step=False, on_epoch=True)
        return {
            "loss": total_loss.detach(),
            "logits": {k: v.detach() for k, v in logits.items()},
            "targets": batch.y,
        }

    def _eval_step(self, phase: str, batch: Batch) -> dict:
        feat = self._hypercolumn_features(batch)
        probe_names = self._active_probe_names()
        logits = self._probe_logits(feat, probe_names)
        losses = self._probe_losses(logits, batch.y)
        total_loss = sum(losses.values()) / max(len(losses), 1)
        self.log(f"{phase}/loss", total_loss, on_step=False, on_epoch=True)
        for name, loss in losses.items():
            self.log(f"{phase}/loss_{name}", loss, on_step=False, on_epoch=True)
        return {"loss": total_loss, "logits": logits, "targets": batch.y}

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        return self._eval_step("val", batch)

    def test_step(self, batch: Batch, batch_idx: int) -> dict:
        return self._eval_step("test", batch)

    def configure_optimizers(self):
        return [
            self.optimizer_factory(params=self.probes[name].parameters(), lr=self.probe_lrs[name])
            for name in self.probe_names
        ]
