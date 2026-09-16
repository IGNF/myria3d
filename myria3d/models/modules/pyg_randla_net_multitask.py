from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, ModuleDict

from myria3d.models.modules.pixel_pooling import pool_and_broadcast_by_cell
from myria3d.models.modules.pyg_randla_net import (
    DilatedResidualBlock,
    FPModule,
    SharedMLP,
    decimate,
)


class PyGRandLANetMultiTask(torch.nn.Module):
    """RandLA-Net with shared backbone and per-task segmentation / regression heads."""

    _TASK_TYPES = frozenset(("semantic", "regression", "pixel_semantic", "tile_distribution"))
    # Task types whose head predicts a class distribution (same Linear(32, num_classes)
    # head as "semantic"); only the pooling/loss applied downstream differs.
    _CLASSIFICATION_LIKE_TYPES = frozenset(("semantic", "pixel_semantic", "tile_distribution"))

    def __init__(
        self,
        num_features: int,
        task_configs: Mapping[str, dict],
        decimation: int = 4,
        num_neighbors: int = 16,
        return_logits: bool = True,
    ):
        super().__init__()
        self.decimation = decimation
        self.return_logits = return_logits
        self.task_configs = {str(k): dict(v) for k, v in task_configs.items()}
        self.tasks = tuple(self.task_configs.keys())

        semantic_num_classes = [
            int(cfg["num_classes"])
            for cfg in self.task_configs.values()
            if cfg.get("task_type", "semantic") in self._CLASSIFICATION_LIKE_TYPES
        ]
        max_num_classes = max(semantic_num_classes) if semantic_num_classes else 1
        d_bottleneck = max(32, max_num_classes, num_features)

        self.fc0 = Linear(num_features, d_bottleneck)
        self.block1 = DilatedResidualBlock(num_neighbors, d_bottleneck, 32)
        self.block2 = DilatedResidualBlock(num_neighbors, 32, 128)
        self.block3 = DilatedResidualBlock(num_neighbors, 128, 256)
        self.block4 = DilatedResidualBlock(num_neighbors, 256, 512)
        self.mlp_summit = SharedMLP([512, 512])
        self.fp4 = FPModule(1, SharedMLP([512 + 256, 256]))
        self.fp3 = FPModule(1, SharedMLP([256 + 128, 128]))
        self.fp2 = FPModule(1, SharedMLP([128 + 32, 32]))
        self.fp1 = FPModule(1, SharedMLP([32 + 32, d_bottleneck]))

        self.mlp_heads = ModuleDict()
        self.fc_heads = ModuleDict()
        for task_name, task_config in self.task_configs.items():
            task_type = self._task_type(task_config)
            self.mlp_heads[task_name] = SharedMLP([d_bottleneck, 64, 32], dropout=[0.0, 0.5])
            if task_type in self._CLASSIFICATION_LIKE_TYPES:
                self.fc_heads[task_name] = Linear(32, int(task_config["num_classes"]))
            else:
                self.fc_heads[task_name] = Linear(32, 1)

    @classmethod
    def _task_type(cls, task_config: dict) -> str:
        task_type = task_config.get("task_type", "semantic")
        if task_type not in cls._TASK_TYPES:
            raise ValueError(
                "Each task_configs entry must set task_type to one of "
                f"{sorted(cls._TASK_TYPES)} (got {task_type!r})."
            )
        return task_type

    def _forward_backbone(self, x, pos, batch, ptr):
        x = x if x is not None else pos

        b1_out = self.block1(self.fc0(x), pos, batch)
        b1_out_decimated, ptr1 = decimate(b1_out, ptr, self.decimation)

        b2_out = self.block2(*b1_out_decimated)
        b2_out_decimated, ptr2 = decimate(b2_out, ptr1, self.decimation)

        b3_out = self.block3(*b2_out_decimated)
        b3_out_decimated, ptr3 = decimate(b3_out, ptr2, self.decimation)

        b4_out = self.block4(*b3_out_decimated)
        b4_out_decimated, _ = decimate(b4_out, ptr3, self.decimation)

        mlp_out = (
            self.mlp_summit(b4_out_decimated[0]),
            b4_out_decimated[1],
            b4_out_decimated[2],
        )

        fp4_out = self.fp4(*mlp_out, *b3_out_decimated)
        fp3_out = self.fp3(*fp4_out, *b2_out_decimated)
        fp2_out = self.fp2(*fp3_out, *b1_out_decimated)
        fp1_out = self.fp1(*fp2_out, *b1_out)
        return fp1_out[0]

    def _forward_encoder_stages(
        self, x, pos, batch, ptr
    ) -> Dict[str, Tuple[Tensor, Tensor, Tensor]]:
        """Returns ``{"enc1": (feat, pos, batch), ..., "enc4": (...)}`` -- the
        block1..block4 outputs, each at its own decimated resolution (enc1 is full
        input resolution; enc2/enc3/enc4 have been decimated 1/2/3 times). Used by
        linear-probing (see ``hypercolumn.py``) to build a multi-scale encoder
        hypercolumn feature; does not touch fp*/decoder and is not used by ``forward``.
        """
        x = x if x is not None else pos

        b1_out = self.block1(self.fc0(x), pos, batch)
        b1_out_decimated, ptr1 = decimate(b1_out, ptr, self.decimation)

        b2_out = self.block2(*b1_out_decimated)
        b2_out_decimated, ptr2 = decimate(b2_out, ptr1, self.decimation)

        b3_out = self.block3(*b2_out_decimated)
        b3_out_decimated, _ = decimate(b3_out, ptr2, self.decimation)

        b4_out = self.block4(*b3_out_decimated)

        return {"enc1": b1_out, "enc2": b2_out, "enc3": b3_out, "enc4": b4_out}

    def forward(
        self,
        x,
        pos,
        batch,
        ptr,
        pooled_head_cell_ids: Optional[Mapping[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """``pooled_head_cell_ids``: optional ``{task_name: per-point cell_id}``, for
        tasks whose ``task_config["pool_before_head"]`` is set. For those tasks, the
        shared backbone feature is pooled to (scene, raster-cell) groups *before* the
        task head runs (mean/max per `task_config["pooling"]`), then broadcast back to
        every point in the cell — matching Pointcept's pixel_semantic recipe (pool
        backbone features, classify once per cell) instead of classifying per-point and
        pooling logits. All downstream code still sees a per-point tensor: every point in
        a cell now simply carries an identical, already-cell-level prediction.
        """
        shared = self._forward_backbone(x, pos, batch, ptr)
        pooled_head_cell_ids = pooled_head_cell_ids or {}
        outputs: Dict[str, Tensor] = {}
        for task_name, task_config in self.task_configs.items():
            features = shared
            cell_id = pooled_head_cell_ids.get(task_name)
            if task_config.get("pool_before_head", False) and cell_id is not None:
                features = pool_and_broadcast_by_cell(
                    shared, cell_id, batch, pooling=task_config.get("pooling", "mean")
                )
            head_features = self.mlp_heads[task_name](features)
            logits = self.fc_heads[task_name](head_features)
            if self._task_type(task_config) == "regression":
                outputs[task_name] = logits.squeeze(-1)
            elif self.return_logits:
                outputs[task_name] = logits
            else:
                outputs[task_name] = logits.log_softmax(dim=-1)
        return outputs

    def last_backbone_layer_parameters(self):
        """Parameters of the last shared layer before any task head runs."""
        return list(self.fp1.parameters())

    def backbone_parameters(self):
        """Parameters of the shared backbone (excludes all task heads)."""
        modules = (
            self.fc0,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.mlp_summit,
            self.fp4,
            self.fp3,
            self.fp2,
            self.fp1,
        )
        return [p for module in modules for p in module.parameters()]

    def task_head_parameters(self, task_name: str):
        """Parameters of a single task's head (mlp_head + fc_head)."""
        return list(self.mlp_heads[task_name].parameters()) + list(
            self.fc_heads[task_name].parameters()
        )
