from typing import Dict, Optional, Tuple

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import knn
from torch_geometric.utils import scatter

from myria3d.models.gradnorm_lite import (
    GradNormLiteEMA,
    combine_weighted_task_losses,
    compute_task_gradient_norms,
    compute_task_last_layer_grad_norms,
    resolve_grad_norm_lite_scales,
)
from myria3d.models.losses import pool_axis_distribution_from_probs
from myria3d.models.model import MODEL_ZOO, get_neural_net_class
from myria3d.models.modules.pixel_pooling import pool_points_by_cell
from myria3d.models.modules.pyg_randla_net_multitask import PyGRandLANetMultiTask
from myria3d.pctl.transforms.transforms import (
    COLOR_FEATURE_NAMES,
    STRENGTH_FEATURE_NAMES,
    resolve_x_feature_names,
)
from myria3d.utils import utils

log = utils.get_logger(__name__)

MODEL_ZOO.append(PyGRandLANetMultiTask)

# Tasks whose loss/metrics are computed by pooling raw (subsampled) per-point predictions
# to a coarser group — a raster cell for "pixel_semantic", a whole tile for
# "tile_distribution" — rather than by KNN-interpolating to the full point cloud.
POOLED_TASK_TYPES = ("pixel_semantic", "tile_distribution")


def _batch_target_key(task_name: str) -> str:
    """Attribute name on `Data`/`Batch` holding a task's raw target: "y" for the main
    (segment) task, "y_{task_name}" for every other task."""
    return "y" if task_name == "segment" else f"y_{task_name}"


def _compute_shared_knn_interpolation_weights(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    k: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KNN neighbor indices and weights once for a fixed geometry pair."""
    with torch.no_grad():
        assign_index = knn(
            pos_x,
            pos_y,
            k,
            batch_x=batch_x,
            batch_y=batch_y,
            num_workers=num_workers,
        )
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
    return y_idx, x_idx, weights


def _apply_knn_interpolation(
    x: torch.Tensor,
    y_idx: torch.Tensor,
    x_idx: torch.Tensor,
    weights: torch.Tensor,
    num_target_points: int,
) -> torch.Tensor:
    """Interpolate features using precomputed KNN indices and weights."""
    y = scatter(x[x_idx] * weights, y_idx, 0, num_target_points, reduce="sum")
    return y / scatter(weights, y_idx, 0, num_target_points, reduce="sum")


class MultiTaskModel(LightningModule):
    """Lightning module for Flair3D+ multitask point cloud learning."""

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["criteria"])

        neural_net_class = get_neural_net_class(kwargs.get("neural_net_class_name"))
        self.model = neural_net_class(**kwargs.get("neural_net_hparams"))

        self.task_configs = kwargs.get("task_configs", {})
        self.task_weights = kwargs.get("task_weights", {})
        self.main_task = kwargs.get("main_task", "segment")
        self.elevation_target_scale = float(kwargs.get("elevation_target_scale", 1.0))

        # Pooled tasks (pixel_semantic, tile_distribution) are always evaluated at the
        # model's native (subsampled) resolution — never KNN-interpolated to the full
        # point cloud, since their targets are inherently coarser than one label per point.
        self._pooled_tasks = tuple(
            name
            for name, cfg in self.task_configs.items()
            if cfg.get("task_type", "semantic") in POOLED_TASK_TYPES
        )
        self._interpolated_tasks = tuple(
            name for name in self.task_configs if name not in self._pooled_tasks
        )

        self.grad_norm_lite_enabled = bool(kwargs.get("grad_norm_lite", False))
        self.grad_norm_lite_interval = int(kwargs.get("grad_norm_lite_interval", 100))
        self.grad_norm_lite_task_groups = dict(kwargs.get("grad_norm_lite_task_groups", {}) or {})
        if self.grad_norm_lite_enabled:
            self._grad_norm_lite_ema = GradNormLiteEMA(
                alpha=float(kwargs.get("grad_norm_lite_ema_alpha", 0.1)),
                eps=float(kwargs.get("grad_norm_lite_eps", 1e-3)),
            )
            self._grad_norm_lite_scales: Dict[str, float] = {}
        self.log_task_gradient_norms_enabled = bool(kwargs.get("log_task_gradient_norms", False))

        self._init_learned_masked_feat(
            enable=bool(kwargs.get("learned_masked_feat", False)),
            keys=kwargs.get("learned_masked_feat_keys", ("color", "strength")),
        )

        criteria_cfg = kwargs.get("criteria", {})
        self.criteria = nn.ModuleDict()
        for task_name in self.task_configs:
            if task_name not in criteria_cfg:
                continue
            criterion_spec = criteria_cfg[task_name]
            if isinstance(criterion_spec, (dict, DictConfig)):
                self.criteria[task_name] = hydra.utils.instantiate(criterion_spec)
            else:
                self.criteria[task_name] = criterion_spec

    def _init_learned_masked_feat(self, enable: bool, keys) -> None:
        """Learned RGB / intensity fill-in for points dropped at train time (Pointcept)."""
        self.learned_masked_feat = enable
        self.learned_masked_feat_keys = tuple(keys or ())
        if not self.learned_masked_feat:
            return
        if "color" in self.learned_masked_feat_keys:
            self.color_mask_value = nn.Parameter(torch.zeros(1, len(COLOR_FEATURE_NAMES)))
        if "strength" in self.learned_masked_feat_keys:
            self.strength_mask_value = nn.Parameter(
                torch.zeros(1, len(STRENGTH_FEATURE_NAMES))
            )

    def _learned_mask_groups(self):
        return (
            ("color", COLOR_FEATURE_NAMES, "color_mask", "color_mask_value"),
            ("strength", STRENGTH_FEATURE_NAMES, "strength_mask", "strength_mask_value"),
        )

    def _fill_masked_features(self, batch: Batch) -> torch.Tensor:
        """Replace dropped RGB / intensity channels with learned fill-in values.

        ``color_mask`` / ``strength_mask`` are True on dropped points. When no mask
        is present (val/test), features are left unchanged. A zero dummy term keeps
        the fill parameters in the autograd graph for DDP when a batch has no drops.
        """
        x = batch.x
        if not self.learned_masked_feat:
            return x

        names = resolve_x_feature_names(batch)
        out = x
        for _, feat_names, mask_key, param_name in self._learned_mask_groups():
            if not hasattr(self, param_name):
                continue
            mask = getattr(batch, mask_key, None)
            if mask is None or not names:
                continue
            present = [name for name in feat_names if name in names]
            if not present:
                continue
            param = getattr(self, param_name).to(dtype=out.dtype, device=out.device)
            if len(names) != out.size(1):
                continue
            point_mask = mask.bool().reshape(-1).to(device=out.device)
            cols = []
            for i, name in enumerate(names):
                col = out[:, i]
                if name in present:
                    fill = param[:, feat_names.index(name)].reshape(())
                    col = torch.where(point_mask, fill, col)
                cols.append(col)
            out = torch.stack(cols, dim=1)

        if self.training:
            for _, _, _, param_name in self._learned_mask_groups():
                if hasattr(self, param_name):
                    out = out + getattr(self, param_name).sum() * 0.0
        return out

    def _log_learned_mask_values(self) -> None:
        if not self.learned_masked_feat:
            return
        for group, feat_names, _, param_name in self._learned_mask_groups():
            if not hasattr(self, param_name):
                continue
            values = getattr(self, param_name).detach().flatten()
            for name, value in zip(feat_names, values):
                self.log(
                    f"train/learned_mask/{group}_{name}",
                    value.item(),
                    on_step=False,
                    on_epoch=True,
                )

    def _task_type(self, task_name: str) -> str:
        return self.task_configs[task_name].get("task_type", "semantic")

    def _get_batch_target(self, batch: Batch, task_name: str) -> Optional[torch.Tensor]:
        return getattr(batch, _batch_target_key(task_name), None)

    def _get_copy_target(self, batch: Batch, task_name: str) -> Optional[torch.Tensor]:
        if "copies" not in batch:
            return None
        copy_key = f"transformed_{_batch_target_key(task_name)}_copy"
        return batch.copies.get(copy_key)

    def _interpolate_outputs(
        self, outputs: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        batch_y = self._get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)
        pos_x = batch.copies["pos_sampled_copy"].cpu()
        pos_y = batch.copies["pos_copy"].cpu()
        y_idx, x_idx, weights = _compute_shared_knn_interpolation_weights(
            pos_x,
            pos_y,
            batch.batch.cpu(),
            batch_y.cpu(),
            k=self.hparams.interpolation_k,
            num_workers=self.hparams.num_workers,
        )
        num_target_points = pos_y.size(0)

        interpolated = {}
        for task_name, preds in outputs.items():
            preds_cpu = preds.cpu()
            if self._task_type(task_name) == "semantic":
                features = preds_cpu
            else:
                features = preds_cpu.unsqueeze(-1)
            result = _apply_knn_interpolation(features, y_idx, x_idx, weights, num_target_points)
            if self._task_type(task_name) != "semantic":
                result = result.squeeze(-1)
            interpolated[task_name] = result
        return interpolated

    def forward(
        self, batch: Batch, *, interpolate: bool = True
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, torch.Tensor]]:
        # Tasks configured with `pool_before_head` (e.g. forest_2d) have their per-task
        # head run on backbone features already pooled to raster cells (Pointcept's
        # pixel_semantic recipe) — see PyGRandLANetMultiTask.forward.
        pooled_head_cell_ids = {
            name: getattr(batch, f"{name}_cell_id")
            for name, cfg in self.task_configs.items()
            if cfg.get("pool_before_head", False)
        }
        raw_outputs = self.model(
            self._fill_masked_features(batch),
            batch.pos,
            batch.batch,
            batch.ptr,
            pooled_head_cell_ids=pooled_head_cell_ids,
        )

        # Pooled tasks (pixel_semantic, tile_distribution) always use the raw, subsampled
        # per-point predictions — their loss/metrics pool these down to a raster cell or a
        # whole tile (see _compute_task_loss), so KNN-interpolating them to the full point
        # cloud first would be both wasted work and, for tile_distribution, incorrect
        # (batch.ptr would no longer match the interpolated point count).
        pooled_outputs = {name: raw_outputs[name] for name in self._pooled_tasks}
        pooled_targets = {
            name: self._get_batch_target(batch, name) for name in self._pooled_tasks
        }

        interp_input = {name: raw_outputs[name] for name in self._interpolated_tasks}
        if self.training or "copies" not in batch or not interpolate:
            interp_outputs = interp_input
            interp_targets = {
                name: self._get_batch_target(batch, name) for name in self._interpolated_tasks
            }
        else:
            interp_outputs = self._interpolate_outputs(interp_input, batch)
            # knn_interpolate runs on CPU (see model.py); copy targets must match output device.
            output_device = (
                next(iter(interp_outputs.values())).device if interp_outputs else batch.pos.device
            )
            interp_targets = {}
            for name in self._interpolated_tasks:
                target = self._get_copy_target(batch, name)
                interp_targets[name] = target.to(output_device) if target is not None else None

        outputs = {**interp_outputs, **pooled_outputs}
        targets = {**interp_targets, **pooled_targets}
        return targets, outputs

    def _compute_pixel_semantic_loss(
        self, task_name: str, criterion, preds: torch.Tensor, targets: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        cell_id = getattr(batch, f"{task_name}_cell_id").to(preds.device)
        batch_index = batch.batch.to(preds.device)
        pooling = self.task_configs[task_name].get("pooling", "mean")
        pooled_preds, pooled_targets, _, _ = pool_points_by_cell(
            preds, cell_id, targets.long(), batch_index, pooling=pooling
        )
        if pooled_preds.size(0) == 0:
            return preds.new_zeros(())
        return criterion(pooled_preds, pooled_targets)

    def _compute_tile_distribution_loss(
        self, task_name: str, criterion, preds: torch.Tensor, targets: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        task_config = self.task_configs[task_name]
        ignore_index = int(task_config["ignore_index"])
        num_classes = int(task_config["num_classes"])
        probs = torch.softmax(preds.float(), dim=-1)
        pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
            probs, targets.long().to(preds.device), batch.ptr.to(preds.device), ignore_index, num_classes
        )
        keep = n_t > 0
        if not bool(keep.any()):
            return preds.new_zeros(())
        return criterion(pi_hat[keep], q_t[keep], weight=n_t[keep])

    def _compute_task_loss(
        self,
        task_name: str,
        preds: torch.Tensor,
        targets: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        criterion = self.criteria[task_name].to(preds.device)
        task_type = self._task_type(task_name)

        if task_type == "pixel_semantic":
            return self._compute_pixel_semantic_loss(task_name, criterion, preds, targets, batch)
        if task_type == "tile_distribution":
            return self._compute_tile_distribution_loss(task_name, criterion, preds, targets, batch)

        targets = targets.to(preds.device)
        if task_type == "semantic":
            return criterion(preds, targets.long())

        valid_mask = torch.isfinite(targets)
        if not valid_mask.any():
            return preds.new_zeros(())
        scaled_preds = preds[valid_mask] / self.elevation_target_scale
        scaled_targets = targets[valid_mask] / self.elevation_target_scale
        return criterion(scaled_preds, scaled_targets)

    def _compute_loss(
        self,
        targets: Dict[str, Optional[torch.Tensor]],
        outputs: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = {}
        total_loss = None
        for task_name, preds in outputs.items():
            task_targets = targets.get(task_name)
            if task_targets is None:
                continue
            task_loss = self._compute_task_loss(task_name, preds, task_targets, batch)
            weight = float(self.task_weights.get(task_name, 1.0))
            weighted_loss = task_loss * weight
            losses[task_name] = task_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        if total_loss is None:
            total_loss = next(iter(outputs.values())).new_zeros(())
        return total_loss, losses

    def _sync_norms_across_ranks(self, norms: Dict[str, float]) -> Dict[str, float]:
        names = sorted(norms.keys())
        if not names:
            return norms
        values = torch.tensor([norms[name] for name in names], device=self.device)
        gathered = self.all_gather(values).float().mean(dim=0)
        return {name: float(gathered[i]) for i, name in enumerate(names)}

    def _apply_grad_norm_lite(self, losses: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Rescale each task's loss by 1/EMA(last-layer grad norm), à la Pointcept's
        GradNorm-lite. The EMA (and the scales derived from it) are refreshed every
        `grad_norm_lite_interval` steps within the current training epoch (Pointcept's
        `engines/train.py` resets its iteration counter every epoch; `batch_idx` mirrors
        that, unlike the run-wide `global_step`); the cached scales are reused for the
        loss combination on every step in between, since recomputing from an unchanged
        EMA would just yield the same values."""
        if batch_idx > 0 and batch_idx % self.grad_norm_lite_interval == 0:
            norms = compute_task_last_layer_grad_norms(
                self.model.last_backbone_layer_parameters(),
                losses,
                self.grad_norm_lite_task_groups,
            )
            if self.trainer is not None and self.trainer.world_size > 1:
                norms = self._sync_norms_across_ranks(norms)
            self._grad_norm_lite_ema.update(norms)
            for name, norm in norms.items():
                self.log(f"train/grad_norm_lite_norm_{name}", norm, on_step=True, on_epoch=False)

            self._grad_norm_lite_scales = resolve_grad_norm_lite_scales(
                self._grad_norm_lite_ema, losses.keys(), self.grad_norm_lite_task_groups
            )
            # One point per group (not per task), matching `norms` above.
            for group in norms.keys():
                self.log(
                    f"train/grad_norm_lite_scale_{group}",
                    self._grad_norm_lite_ema.scale(group),
                    on_step=True,
                    on_epoch=False,
                )

        total_loss, _ = combine_weighted_task_losses(
            losses, self.task_weights, self._grad_norm_lite_scales
        )
        return total_loss

    def _log_task_gradient_norms(self, losses: Dict[str, torch.Tensor]) -> None:
        """Diagnostic-only: log per-task backbone/head grad norms and pairwise
        backbone-gradient cosine similarities between tasks. Does not affect training."""
        result = compute_task_gradient_norms(
            self.model.backbone_parameters(),
            {task_name: self.model.task_head_parameters(task_name) for task_name in losses},
            losses,
            self.task_weights,
        )
        for task_name, norm in result["norms"].items():
            self.log(
                f"train/task_grad_norm_backbone_{task_name}",
                norm["backbone"],
                on_step=True,
                on_epoch=False,
            )
            self.log(
                f"train/task_grad_norm_head_{task_name}",
                norm["head"],
                on_step=True,
                on_epoch=False,
            )
        for pair_name, cos in result["backbone_cos"].items():
            self.log(f"train/task_grad_cos_{pair_name}", cos, on_step=True, on_epoch=False)

    def training_step(self, batch: Batch, batch_idx: int) -> dict:
        targets, outputs = self.forward(batch)
        loss, losses = self._compute_loss(targets, outputs, batch)
        if self.grad_norm_lite_enabled:
            loss = self._apply_grad_norm_lite(losses, batch_idx)
        if self.log_task_gradient_norms_enabled:
            self._log_task_gradient_norms(losses)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        for task_name, task_loss in losses.items():
            self.log(f"train/loss_{task_name}", task_loss, on_step=False, on_epoch=True)
        self._log_learned_mask_values()
        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def validation_step(self, batch: Batch, batch_idx: int) -> dict:
        interpolate = bool(getattr(self.hparams, "interpolate_at_val", True))
        targets, outputs = self.forward(batch, interpolate=interpolate)
        loss, losses = self._compute_loss(targets, outputs, batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True)
        for task_name, task_loss in losses.items():
            self.log(f"val/loss_{task_name}", task_loss, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def test_step(self, batch: Batch, batch_idx: int):
        interpolate = bool(getattr(self.hparams, "interpolate_at_test", True))
        targets, outputs = self.forward(batch, interpolate=interpolate)
        loss, losses = self._compute_loss(targets, outputs, batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        for task_name, task_loss in losses.items():
            self.log(f"test/loss_{task_name}", task_loss, on_step=False, on_epoch=True)
        return {
            "loss": loss,
            "outputs": outputs,
            "targets": targets,
        }

    def predict_step(self, batch: Batch) -> dict:
        _, outputs = self.forward(batch)
        return {"outputs": {k: v.detach().cpu() for k, v in outputs.items()}}

    def configure_optimizers(self):
        self.lr = self.hparams.lr
        optimizer = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        if self.hparams.lr_scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.hparams.lr_scheduler(optimizer),
                "monitor": self.hparams.monitor,
                "interval": "epoch",
                "frequency": self.hparams.get("lr_scheduler_frequency", 1),
            },
        }

    def _get_batch_tensor_by_enumeration(self, pos_x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.full((len(sample_pos),), i) for i, sample_pos in enumerate(pos_x)])
