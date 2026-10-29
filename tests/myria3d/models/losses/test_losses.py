import pytest
import torch
import torch.nn.functional as F

from myria3d.models.losses.combined import CombinedLoss
from myria3d.models.losses.focal import FocalLoss
from myria3d.models.losses.lovasz import LovaszLoss

"""
Unit tests for the custom loss functions (focal, Lovász and their combination).

These tests work on small, hand-crafted point-cloud-like tensors with the layout used
throughout the project: logits of shape ``(P, C)`` (P points, C classes) and integer
targets of shape ``(P,)``.

"""

NUM_CLASSES = 7
NUM_POINTS = 1024


@pytest.fixture
def logits_and_targets():
    """A reproducible batch of logits ``(P, C)`` and integer targets ``(P,)``."""
    generator = torch.Generator().manual_seed(0)
    logits = torch.randn(NUM_POINTS, NUM_CLASSES, generator=generator)
    targets = torch.randint(0, NUM_CLASSES, (NUM_POINTS,), generator=generator)
    return logits, targets


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------


def test_focal_loss_returns_finite_scalar(logits_and_targets):
    logits, targets = logits_and_targets
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_loss_gamma_zero_matches_cross_entropy(logits_and_targets):
    """With ``gamma=0`` the focal factor is 1, so focal loss == cross-entropy."""
    logits, targets = logits_and_targets
    focal = FocalLoss(gamma=0.0)(logits, targets)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_focal_loss_downweights_easy_examples(logits_and_targets):
    """Focal loss (gamma>0) should be smaller than plain cross-entropy here,
    since it down-weights well-classified points."""
    logits, targets = logits_and_targets
    focal = FocalLoss(gamma=2.0)(logits, targets)
    ce = F.cross_entropy(logits, targets)
    assert focal.item() < ce.item()


def test_focal_loss_ignore_index():
    """Points with the ignore index must not contribute to the loss."""
    logits = torch.randn(6, NUM_CLASSES)
    targets = torch.tensor([0, 1, 2, 3, 0, 1])
    ignored = targets.clone()
    ignored[:2] = -100  # ignore first two points

    full = FocalLoss(gamma=2.0, ignore_index=-100)(logits, targets)
    partial = FocalLoss(gamma=2.0, ignore_index=-100)(logits[2:], targets[2:])
    masked = FocalLoss(gamma=2.0, ignore_index=-100)(logits, ignored)
    assert torch.allclose(masked, partial, atol=1e-6)
    assert not torch.allclose(masked, full, atol=1e-6)


def test_focal_loss_all_ignored_is_zero():
    logits = torch.randn(4, NUM_CLASSES)
    targets = torch.full((4,), -100)
    loss = FocalLoss(gamma=2.0, ignore_index=-100)(logits, targets)
    assert loss.item() == 0.0


def test_focal_loss_with_alpha_weights(logits_and_targets):
    logits, targets = logits_and_targets
    alpha = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    loss = FocalLoss(gamma=2.0, alpha=alpha)(logits, targets)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_focal_loss_reductions(logits_and_targets, reduction):
    logits, targets = logits_and_targets
    loss = FocalLoss(gamma=2.0, reduction=reduction)(logits, targets)
    if reduction == "none":
        assert loss.shape == (NUM_POINTS,)
    else:
        assert loss.ndim == 0


def test_focal_loss_invalid_reduction_raises():
    with pytest.raises(ValueError):
        FocalLoss(reduction="invalid")


def test_focal_loss_backward(logits_and_targets):
    logits, targets = logits_and_targets
    logits = logits.clone().requires_grad_(True)
    FocalLoss(gamma=2.0)(logits, targets).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# LovaszLoss
# ---------------------------------------------------------------------------


def test_lovasz_multiclass_returns_finite_scalar(logits_and_targets):
    logits, targets = logits_and_targets
    loss = LovaszLoss(mode="multiclass")(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_lovasz_perfect_prediction_is_low():
    """Confident, correct logits should yield a near-zero Lovász loss."""
    targets = torch.arange(NUM_CLASSES).repeat(4)
    logits = F.one_hot(targets, NUM_CLASSES).float() * 20.0  # very confident
    loss = LovaszLoss(mode="multiclass")(logits, targets)
    assert loss.item() < 1e-2


def test_lovasz_ignore_index():
    targets = torch.arange(NUM_CLASSES).repeat(4)
    logits = torch.randn(targets.numel(), NUM_CLASSES)
    with_ignore = LovaszLoss(mode="multiclass", ignore_index=0)(logits, targets)
    assert torch.isfinite(with_ignore)


def test_lovasz_loss_weight_scales():
    targets = torch.arange(NUM_CLASSES).repeat(4)
    logits = torch.randn(targets.numel(), NUM_CLASSES)
    base = LovaszLoss(mode="multiclass", loss_weight=1.0)(logits, targets)
    scaled = LovaszLoss(mode="multiclass", loss_weight=2.5)(logits, targets)
    assert torch.allclose(scaled, 2.5 * base, atol=1e-6)


def test_lovasz_invalid_mode_raises():
    with pytest.raises(AssertionError):
        LovaszLoss(mode="not-a-mode")


def test_lovasz_backward():
    targets = torch.arange(NUM_CLASSES).repeat(4)
    logits = torch.randn(targets.numel(), NUM_CLASSES, requires_grad=True)
    LovaszLoss(mode="multiclass")(logits, targets).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------


def test_combined_loss_default_weights_sum(logits_and_targets):
    logits, targets = logits_and_targets
    focal = FocalLoss(gamma=2.0)
    lovasz = LovaszLoss(mode="multiclass")
    combined = CombinedLoss(losses=[focal, lovasz])
    expected = focal(logits, targets) + lovasz(logits, targets)
    assert torch.allclose(combined(logits, targets), expected, atol=1e-6)


def test_combined_loss_custom_weights(logits_and_targets):
    logits, targets = logits_and_targets
    focal = FocalLoss(gamma=2.0)
    lovasz = LovaszLoss(mode="multiclass")
    weights = [0.7, 0.3]
    combined = CombinedLoss(losses=[focal, lovasz], weights=weights)
    expected = weights[0] * focal(logits, targets) + weights[1] * lovasz(logits, targets)
    assert torch.allclose(combined(logits, targets), expected, atol=1e-6)


def test_combined_loss_mismatched_weights_raises():
    with pytest.raises(AssertionError):
        CombinedLoss(losses=[FocalLoss(), LovaszLoss(mode="multiclass")], weights=[1.0])


def test_combined_loss_single_loss(logits_and_targets):
    logits, targets = logits_and_targets
    focal = FocalLoss(gamma=2.0)
    combined = CombinedLoss(losses=[focal], weights=[2.0])
    assert torch.allclose(combined(logits, targets), 2.0 * focal(logits, targets), atol=1e-6)


def test_combined_loss_backward(logits_and_targets):
    logits, targets = logits_and_targets
    logits = logits.clone().requires_grad_(True)
    combined = CombinedLoss(
        losses=[FocalLoss(gamma=2.0), LovaszLoss(mode="multiclass")],
        weights=[0.5, 0.5],
    )
    combined(logits, targets).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
