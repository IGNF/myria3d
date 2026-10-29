from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Multiclass Focal loss for segmentation tasks.

    Focal loss down-weights well-classified examples and focuses training on
    hard, misclassified points. See Lin et al., "Focal Loss for Dense Object
    Detection" (https://arxiv.org/abs/1708.02002).

    The loss is computed as::

        FL(p_t) = -alpha_t * (1 - p_t) ** gamma * log(p_t)

    Args:
        gamma: Focusing parameter. ``gamma=0`` is equivalent to cross-entropy.
        alpha: Optional per-class weighting factor (tensor of shape ``[C]``).
        ignore_index: Target value that is ignored and does not contribute to
            the loss.
        reduction: One of ``"mean"``, ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float)
        self.register_buffer("alpha", alpha)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Shape:
            - y_pred: torch.Tensor of shape (N, C) or (N, C, d1, d2, ...)
            - y_true: torch.Tensor of shape (N,) or (N, d1, d2, ...)
        """
        C = y_pred.size(1)
        if y_pred.dim() > 2:
            # (N, C, d1, d2, ...) -> (P, C)
            y_pred = torch.movedim(y_pred, 1, -1).contiguous().view(-1, C)
        y_true = y_true.view(-1)

        alpha = self.alpha
        if alpha is not None:
            alpha = alpha.to(dtype=y_pred.dtype, device=y_pred.device)

        log_prob = F.log_softmax(y_pred, dim=-1)
        ce_loss = F.nll_loss(
            log_prob,
            y_true,
            weight=alpha,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        valid = y_true != self.ignore_index
        # Probability assigned to the ground-truth class for each point.
        safe_true = y_true.clone()
        safe_true[~valid] = 0
        pt = log_prob.gather(1, safe_true.unsqueeze(1)).squeeze(1).exp()
        focal_factor = (1.0 - pt) ** self.gamma

        loss = focal_factor * ce_loss
        loss = loss[valid]

        if self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else y_pred.new_zeros(())
        if self.reduction == "sum":
            return loss.sum()
        return loss
