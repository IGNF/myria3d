from typing import Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning import Callback
from torchmetrics import JaccardIndex, MeanAbsoluteError, MeanSquaredError

from myria3d.models.dilated_metrics import (
    dilated_prf_enabled,
    dilated_precision_recall_counts,
    precision_recall_f1,
)
from myria3d.models.losses import (
    abs_freq_error_rows,
    kl_divergence_rows,
    pool_axis_distribution_from_probs,
)
from myria3d.models.modules.pixel_pooling import pool_points_by_cell
from myria3d.pctl.dataset.flair3d_label_remap import (
    NATURAL_HABITAT_AXIS_DEFINITIONS,
    get_definition,
)

# pixel_semantic IoU is computed after pooling to raster cells (same grouping as the
# training loss). That matches Pointcept's dense-raster mIoU, not a point-density proxy.
_IOU_TRACKED_TASK_TYPES = ("semantic", "pixel_semantic")
_PHASES = ("train", "val", "test")
_PIXEL_SEMANTIC_FALLBACK_NAMES = {
    "forest_2d": ["Not Forest", "Forest"],
    "roads": ["Background", "Road"],
}


def class_name_slug(class_name: str) -> str:
    """Sanitize a class name for metric tags (Pointcept ``class_name_slug``)."""
    return "".join(
        c if (c.isalnum() or c in "._-") else "_"
        for c in str(class_name).strip().replace(" ", "_")
    )


def metric_tag(split: str, name: str, task: Optional[str] = None) -> str:
    if task is None:
        return f"{split}/{name}"
    return f"{split}/{task}/{name}"


def iou_class_tag(split: str, slug: str, task: Optional[str] = None) -> str:
    if task is None:
        return f"{split}/iou/{slug}"
    return f"{split}/{task}/iou/{slug}"


def _class_names_for_task(
    task_name: str,
    task_config: dict,
    classification_dict: Optional[Dict[int, str]] = None,
) -> List[str]:
    names = task_config.get("names")
    if names:
        return [str(n) for n in names]
    if task_name == "segment" and classification_dict:
        return [
            classification_dict[int(class_id)]
            for class_id in sorted(classification_dict.keys(), key=int)
        ]
    if task_name in NATURAL_HABITAT_AXIS_DEFINITIONS:
        defn = get_definition("natural_habitat", NATURAL_HABITAT_AXIS_DEFINITIONS[task_name])
        return list(defn.names)[: int(task_config["num_classes"])]
    if task_name in _PIXEL_SEMANTIC_FALLBACK_NAMES:
        return list(_PIXEL_SEMANTIC_FALLBACK_NAMES[task_name])
    return [str(class_id) for class_id in range(int(task_config["num_classes"]))]


class MultiTaskMetrics(Callback):
    """Per-task metrics aligned with Pointcept Flair3D+ multitask logging."""

    def __init__(
        self,
        task_configs: dict,
        main_task: str = "segment",
        elevation_target_scale: float = 1.0,
        classification_dict: Optional[Dict[int, str]] = None,
    ):
        self.task_configs = task_configs
        self.main_task = main_task
        self.elevation_target_scale = elevation_target_scale
        self.classification_dict = classification_dict or {}
        self.best_val_iou = 0.0

        self.semantic_iou: Dict[str, Dict[str, JaccardIndex]] = {
            phase: {} for phase in _PHASES
        }
        # Per-class IoU is only logged at test time (val stays macro-only to keep the
        # wandb dashboard readable over a long multitask run).
        self.semantic_iou_by_class: Dict[str, Dict[str, JaccardIndex]] = {
            phase: {} for phase in ("test",)
        }
        self.class_names_by_task: Dict[str, List[str]] = {}
        for task_name, task_config in task_configs.items():
            if task_config.get("task_type", "semantic") not in _IOU_TRACKED_TASK_TYPES:
                continue
            num_classes = int(task_config["num_classes"])
            ignore_index = int(task_config.get("ignore_index", num_classes - 1))
            # torchmetrics JaccardIndex indexes `ignore_index` into a num_classes
            # confusion matrix. pixel_semantic void (2) is outside {0, 1} — filter
            # those labels before update instead of passing an OOB ignore_index.
            tm_ignore = ignore_index if 0 <= ignore_index < num_classes else None
            self.class_names_by_task[task_name] = _class_names_for_task(
                task_name, task_config, self.classification_dict
            )
            for phase in _PHASES:
                self.semantic_iou[phase][task_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                    ignore_index=tm_ignore,
                )
            for phase in self.semantic_iou_by_class:
                self.semantic_iou_by_class[phase][task_name] = JaccardIndex(
                    task="multiclass",
                    num_classes=num_classes,
                    average=None,
                    ignore_index=tm_ignore,
                )

        self.elevation_mae = {phase: MeanAbsoluteError() for phase in _PHASES}
        self.elevation_rmse = {phase: MeanSquaredError(squared=False) for phase in _PHASES}
        self._elevation_updated = {phase: False for phase in _PHASES}

        self.tile_distribution_tasks: List[str] = [
            task_name
            for task_name, task_config in task_configs.items()
            if task_config.get("task_type", "semantic") == "tile_distribution"
        ]
        self.pixel_semantic_tasks: List[str] = [
            task_name
            for task_name, task_config in task_configs.items()
            if task_config.get("task_type", "semantic") == "pixel_semantic"
        ]
        zeros = {phase: {t: 0.0 for t in self.pixel_semantic_tasks} for phase in _PHASES}
        self._prf_tp = {phase: dict(d) for phase, d in zeros.items()}
        self._prf_fp = {phase: dict(d) for phase, d in zeros.items()}
        self._prf_fn = {phase: dict(d) for phase, d in zeros.items()}
        self._dilated_p_num = {phase: dict(d) for phase, d in zeros.items()}
        self._dilated_p_denom = {phase: dict(d) for phase, d in zeros.items()}
        self._dilated_r_num = {phase: dict(d) for phase, d in zeros.items()}
        self._dilated_r_denom = {phase: dict(d) for phase, d in zeros.items()}
        self._kl_weighted_sum: Dict[str, Dict[str, float]] = {
            phase: {task_name: 0.0 for task_name in self.tile_distribution_tasks}
            for phase in _PHASES
        }
        self._kl_weight_sum: Dict[str, Dict[str, float]] = {
            phase: {task_name: 0.0 for task_name in self.tile_distribution_tasks}
            for phase in _PHASES
        }
        self._td_abs_weighted: Dict[str, Dict[str, torch.Tensor]] = {
            phase: {
                task_name: torch.zeros(
                    int(task_configs[task_name]["num_classes"]), dtype=torch.float64
                )
                for task_name in self.tile_distribution_tasks
            }
            for phase in _PHASES
        }
        for task_name in self.tile_distribution_tasks:
            if task_name not in self.class_names_by_task:
                self.class_names_by_task[task_name] = _class_names_for_task(
                    task_name, task_configs[task_name], self.classification_dict
                )

    def _iou_pred_target(self, task_name: str, preds, targets, batch):
        """Argmax labels (and matching targets), pooling pixel_semantic tasks to cells.

        Returns ``(pred_labels, targets, cell_id, batch_index)``. ``cell_id`` /
        ``batch_index`` are None for per-point semantic tasks.
        """
        task_type = self.task_configs[task_name].get("task_type", "semantic")
        if task_type == "pixel_semantic" and batch is not None:
            cell_id = getattr(batch, f"{task_name}_cell_id", None)
            if cell_id is not None:
                pooling = self.task_configs[task_name].get("pooling", "mean")
                pooled_preds, pooled_targets, pooled_cell_id, pooled_batch = (
                    pool_points_by_cell(
                        preds.detach(),
                        cell_id.to(preds.device),
                        targets.long().to(preds.device),
                        batch.batch.to(preds.device),
                        pooling=pooling,
                    )
                )
                if pooled_preds.size(0) == 0:
                    return None, None, None, None
                return (
                    torch.argmax(pooled_preds, dim=1),
                    pooled_targets,
                    pooled_cell_id,
                    pooled_batch,
                )
        pred_labels = torch.argmax(preds.detach(), dim=1)
        return pred_labels, targets.long().to(pred_labels.device), None, None

    @staticmethod
    def _graph_hw(batch, task_name: str, graph_idx: int) -> tuple:
        height = getattr(batch, f"{task_name}_raster_h", None)
        width = getattr(batch, f"{task_name}_raster_w", None)
        if height is None or width is None:
            return 0, 0
        return int(height.reshape(-1)[graph_idx].item()), int(
            width.reshape(-1)[graph_idx].item()
        )

    def _update_pixel_prf(
        self,
        phase: str,
        task_name: str,
        pred_labels,
        targets,
        cell_id,
        batch_index,
        batch,
    ) -> None:
        task_config = self.task_configs[task_name]
        ignore_index = int(task_config.get("ignore_index", -1))
        fg_index = int(task_config.get("fg_index", 1))
        valid = targets != ignore_index
        pred_fg = (pred_labels == fg_index) & valid
        gt_fg = (targets == fg_index) & valid
        self._prf_tp[phase][task_name] += float((pred_fg & gt_fg).sum().item())
        self._prf_fp[phase][task_name] += float((pred_fg & ~gt_fg).sum().item())
        self._prf_fn[phase][task_name] += float((~pred_fg & gt_fg).sum().item())

        if (
            cell_id is None
            or batch is None
            or not dilated_prf_enabled(task_config)
        ):
            return
        radius_px = int(task_config.get("buffer_radius_px", 3))
        pred_np = pred_labels.detach().cpu().numpy()
        tgt_np = targets.detach().cpu().numpy()
        cid_np = cell_id.detach().cpu().numpy().astype(np.int64)
        b_np = batch_index.detach().cpu().numpy().astype(np.int64)
        for graph_idx in np.unique(b_np):
            height, width = self._graph_hw(batch, task_name, int(graph_idx))
            if height <= 0 or width <= 0:
                continue
            sel = b_np == graph_idx
            rows = cid_np[sel] // width
            cols = cid_np[sel] % width
            in_grid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            scored = in_grid & (tgt_np[sel] != ignore_index)
            pred_grid = np.zeros((height, width), dtype=bool)
            gt_grid = np.zeros((height, width), dtype=bool)
            valid_grid = np.zeros((height, width), dtype=bool)
            rr, cc = rows[scored], cols[scored]
            valid_grid[rr, cc] = True
            pred_grid[rr, cc] = pred_np[sel][scored] == fg_index
            gt_grid[rr, cc] = tgt_np[sel][scored] == fg_index
            p_num, p_denom, r_num, r_denom = dilated_precision_recall_counts(
                pred_grid, gt_grid, valid_grid, radius_px=radius_px
            )
            self._dilated_p_num[phase][task_name] += p_num
            self._dilated_p_denom[phase][task_name] += p_denom
            self._dilated_r_num[phase][task_name] += r_num
            self._dilated_r_denom[phase][task_name] += r_denom

    def _end_of_batch(self, phase: str, outputs: dict, batch=None):
        preds_by_task = outputs["outputs"]
        targets_by_task = outputs["targets"]

        for task_name, metric in self.semantic_iou[phase].items():
            preds = preds_by_task.get(task_name)
            targets = targets_by_task.get(task_name)
            if preds is None or targets is None:
                continue
            pred_labels, metric_targets, cell_id, batch_index = self._iou_pred_target(
                task_name, preds, targets, batch
            )
            if pred_labels is None:
                continue
            ignore_index = int(self.task_configs[task_name].get("ignore_index", -1))
            if ignore_index >= 0:
                iou_keep = metric_targets != ignore_index
                iou_pred, iou_tgt = pred_labels[iou_keep], metric_targets[iou_keep]
            else:
                iou_pred, iou_tgt = pred_labels, metric_targets
            if iou_pred.numel() > 0:
                metric.to(iou_pred.device)(iou_pred, iou_tgt)
                by_class = self.semantic_iou_by_class.get(phase, {}).get(task_name)
                if by_class is not None:
                    by_class.to(iou_pred.device)(iou_pred, iou_tgt)
            if task_name in self.pixel_semantic_tasks:
                self._update_pixel_prf(
                    phase,
                    task_name,
                    pred_labels,
                    metric_targets,
                    cell_id,
                    batch_index,
                    batch,
                )

        for task_name in self.tile_distribution_tasks:
            preds = preds_by_task.get(task_name)
            targets = targets_by_task.get(task_name)
            if preds is None or targets is None or batch is None:
                continue
            task_config = self.task_configs[task_name]
            ignore_index = int(task_config["ignore_index"])
            num_classes = int(task_config["num_classes"])
            probs = torch.softmax(preds.detach().float(), dim=-1)
            ptr = batch.ptr.to(probs.device)
            targets = targets.long().to(probs.device)
            pi_hat, q_t, n_t = pool_axis_distribution_from_probs(
                probs, targets, ptr, ignore_index, num_classes
            )
            keep = n_t > 0
            if not bool(keep.any()):
                continue
            kl = kl_divergence_rows(q_t[keep], pi_hat[keep])
            weight = n_t[keep]
            self._kl_weighted_sum[phase][task_name] += float((kl * weight).sum().item())
            self._kl_weight_sum[phase][task_name] += float(weight.sum().item())
            abs_err = abs_freq_error_rows(pi_hat[keep], q_t[keep])
            self._td_abs_weighted[phase][task_name] += (
                (weight.unsqueeze(-1) * abs_err).sum(0).detach().cpu().double()
            )

        if "elevation" in preds_by_task and targets_by_task.get("elevation") is not None:
            preds = preds_by_task["elevation"].detach()
            targets = targets_by_task["elevation"].to(preds.device)
            valid_mask = torch.isfinite(preds) & torch.isfinite(targets)
            if valid_mask.any():
                # Metrics in meters (unscaled). Loss uses elevation_target_scale separately.
                self.elevation_mae[phase].to(preds.device)(
                    preds[valid_mask], targets[valid_mask]
                )
                self.elevation_rmse[phase].to(preds.device)(
                    preds[valid_mask], targets[valid_mask]
                )
                self._elevation_updated[phase] = True

    def _log_miou(self, phase: str, task_name: str, value, pl_module) -> None:
        pl_module.log(
            metric_tag(phase, "mIoU", task=task_name),
            value,
            on_epoch=True,
            on_step=False,
        )
        # Aliases used by ModelCheckpoint / early stopping (`val/iou` on segment).
        alias = f"{phase}/iou" if task_name == self.main_task else f"{phase}/iou_{task_name}"
        pl_module.log(alias, value, on_epoch=True, on_step=False)

    def _end_of_epoch(self, phase: str, pl_module):
        for task_name, metric in self.semantic_iou[phase].items():
            value = metric.to(pl_module.device).compute()
            self._log_miou(phase, task_name, value, pl_module)
            if phase == "val" and task_name == self.main_task:
                self.best_val_iou = max(self.best_val_iou, float(value))
                pl_module.log(
                    "val/best_iou",
                    self.best_val_iou,
                    on_epoch=True,
                    on_step=False,
                    metric_attribute="val/best_iou",
                )
                pl_module.log(
                    metric_tag("val", "mIoU_best", task=task_name),
                    self.best_val_iou,
                    on_epoch=True,
                    on_step=False,
                )
            metric.reset()

        if phase in self.semantic_iou_by_class:
            for task_name, metric in self.semantic_iou_by_class[phase].items():
                values = metric.to(pl_module.device).compute()
                ignore_index = int(self.task_configs[task_name].get("ignore_index", -1))
                names = self.class_names_by_task[task_name]
                for class_idx, (value, class_name) in enumerate(zip(values, names)):
                    if class_idx == ignore_index or not torch.isfinite(value):
                        continue
                    pl_module.log(
                        iou_class_tag(phase, class_name_slug(class_name), task=task_name),
                        value,
                        on_epoch=True,
                        on_step=False,
                    )
                metric.reset()

        for task_name in self.pixel_semantic_tasks:
            tp = self._prf_tp[phase][task_name]
            fp = self._prf_fp[phase][task_name]
            fn = self._prf_fn[phase][task_name]
            if tp + fp + fn > 0:
                precision, recall, f1 = precision_recall_f1(tp, tp + fp, tp, tp + fn)
                pl_module.log(
                    metric_tag(phase, "precision", task=task_name),
                    precision,
                    on_epoch=True,
                    on_step=False,
                )
                pl_module.log(
                    metric_tag(phase, "recall", task=task_name),
                    recall,
                    on_epoch=True,
                    on_step=False,
                )
                pl_module.log(
                    metric_tag(phase, "f1", task=task_name),
                    f1,
                    on_epoch=True,
                    on_step=False,
                )
            self._prf_tp[phase][task_name] = 0.0
            self._prf_fp[phase][task_name] = 0.0
            self._prf_fn[phase][task_name] = 0.0

            if dilated_prf_enabled(self.task_configs[task_name]):
                p_num = self._dilated_p_num[phase][task_name]
                p_denom = self._dilated_p_denom[phase][task_name]
                r_num = self._dilated_r_num[phase][task_name]
                r_denom = self._dilated_r_denom[phase][task_name]
                if p_denom + r_denom > 0:
                    d_p, d_r, d_f1 = precision_recall_f1(p_num, p_denom, r_num, r_denom)
                    pl_module.log(
                        metric_tag(phase, "dilated_precision", task=task_name),
                        d_p,
                        on_epoch=True,
                        on_step=False,
                    )
                    pl_module.log(
                        metric_tag(phase, "dilated_recall", task=task_name),
                        d_r,
                        on_epoch=True,
                        on_step=False,
                    )
                    pl_module.log(
                        metric_tag(phase, "dilated_f1", task=task_name),
                        d_f1,
                        on_epoch=True,
                        on_step=False,
                    )
                self._dilated_p_num[phase][task_name] = 0.0
                self._dilated_p_denom[phase][task_name] = 0.0
                self._dilated_r_num[phase][task_name] = 0.0
                self._dilated_r_denom[phase][task_name] = 0.0

        if self._elevation_updated[phase]:
            mae = self.elevation_mae[phase].compute()
            rmse = self.elevation_rmse[phase].compute()
            pl_module.log(
                metric_tag(phase, "mae", task="reg/elevation"),
                mae,
                on_epoch=True,
                on_step=False,
            )
            pl_module.log(
                metric_tag(phase, "rmse", task="reg/elevation"),
                rmse,
                on_epoch=True,
                on_step=False,
            )
            # Short aliases kept for existing dashboards.
            pl_module.log(f"{phase}/elevation_mae", mae, on_epoch=True, on_step=False)
            pl_module.log(f"{phase}/elevation_rmse", rmse, on_epoch=True, on_step=False)
            self.elevation_mae[phase].reset()
            self.elevation_rmse[phase].reset()
            self._elevation_updated[phase] = False

        kl_by_task = {}
        tv_by_task = {}
        for task_name in self.tile_distribution_tasks:
            weight_sum = self._kl_weight_sum[phase][task_name]
            if weight_sum > 0:
                kl_value = self._kl_weighted_sum[phase][task_name] / weight_sum
                mae = self._td_abs_weighted[phase][task_name] / weight_sum
                tv_value = float(mae.sum().item())
                kl_by_task[task_name] = kl_value
                tv_by_task[task_name] = tv_value
                pl_module.log(
                    f"{phase}/weighted_kl/{task_name}", kl_value, on_epoch=True, on_step=False
                )
                pl_module.log(f"{phase}/tv/{task_name}", tv_value, on_epoch=True, on_step=False)
                pl_module.log(
                    f"{phase}/kl_{task_name}", kl_value, on_epoch=True, on_step=False
                )
                names = self.class_names_by_task.get(task_name, [])
                for class_idx, class_mae in enumerate(mae.tolist()):
                    slug = (
                        class_name_slug(names[class_idx])
                        if class_idx < len(names)
                        else str(class_idx)
                    )
                    pl_module.log(
                        f"{phase}/mae/{task_name}/{slug}",
                        class_mae,
                        on_epoch=True,
                        on_step=False,
                    )
            self._kl_weighted_sum[phase][task_name] = 0.0
            self._kl_weight_sum[phase][task_name] = 0.0
            self._td_abs_weighted[phase][task_name].zero_()

        if kl_by_task:
            nathab_kl = [v for k, v in kl_by_task.items() if k.startswith("nathab_")]
            nathab_tv = [v for k, v in tv_by_task.items() if k.startswith("nathab_")]
            if nathab_kl:
                pl_module.log(
                    f"{phase}/weighted_kl/nathab_total",
                    float(sum(nathab_kl)),
                    on_epoch=True,
                    on_step=False,
                )
                pl_module.log(
                    f"{phase}/tv/nathab_total",
                    float(sum(nathab_tv)),
                    on_epoch=True,
                    on_step=False,
                )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("train", outputs, batch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("val", outputs, batch)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end_of_batch("test", outputs, batch)

    def on_train_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("train", pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("val", pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self._end_of_epoch("test", pl_module)
