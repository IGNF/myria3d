"""Multi-scale encoder hypercolumn for linear probing on a frozen RandLA-Net backbone."""

from typing import Dict, Sequence, Tuple

import torch
from torch import Tensor
from torch_geometric.nn.unpool import knn_interpolate

# Finest-first, matching Pointcept's hypercolumn concat convention. Channel widths per
# stage, fixed by PyGRandLANetMultiTask.__init__'s block1-4 architecture.
STAGE_CHANNELS: Dict[str, int] = {"enc1": 32, "enc2": 128, "enc3": 256, "enc4": 512}
DEFAULT_SCALES: Tuple[str, ...] = ("enc1", "enc2", "enc3", "enc4")


def hypercolumn_concat(
    stage_feats: Dict[str, Tuple[Tensor, Tensor, Tensor]],
    target_pos: Tensor,
    target_batch: Tensor,
    scales: Sequence[str] = DEFAULT_SCALES,
    k: int = 1,
) -> Tensor:
    """Concatenate per-point features from several encoder stages at ``target_pos``
    resolution (finest first).

    Grid/voxel-pooling backbones (Pointcept's PTv3/SpUNet) track a pooling
    parent/child index and gather hypercolumn features with it -- no spatial
    interpolation needed. RandLA-Net's ``decimate()`` is a random per-cloud
    subsample with no such index, so this uses ``knn_interpolate`` (the same
    primitive ``FPModule`` uses to upsample between stages) to bring every
    requested stage back to ``target_pos``/``target_batch`` before concatenating.
    """
    parts = []
    for scale in scales:
        feat, pos, batch = stage_feats[scale]
        parts.append(knn_interpolate(feat, pos, target_pos, batch, target_batch, k=k))
    return torch.cat(parts, dim=1)
