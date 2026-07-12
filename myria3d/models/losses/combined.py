from typing import List, Optional

import torch
from torch import nn


class CombinedLoss(nn.Module):
    """Weighted sum of several loss functions.

    Each sub-loss is called as ``loss(logits, targets)`` and the results are
    combined as ``sum(weight_i * loss_i)``. This allows, for instance, to train
    with a weighted cross-entropy loss and a Lovász loss at the same time.
    """

    def __init__(self, losses: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        if weights is None:
            weights = [1.0] * len(self.losses)
        assert len(weights) == len(self.losses), (
            "The number of weights must match the number of losses "
            f"({len(weights)} weights for {len(self.losses)} losses)."
        )
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = logits.new_zeros(())
        for weight, loss in zip(self.weights, self.losses):
            total_loss = total_loss + weight * loss(logits, targets)
        return total_loss
