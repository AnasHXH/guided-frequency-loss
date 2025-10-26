"""
Unit tests for loss functions.
"""

import torch
import pytest
from gfl.losses import GradualFrequencyLoss, CharbonnierLoss


def test_gradual_frequency_loss():
    """Test GradualFrequencyLoss computation."""
    loss_fn = GradualFrequencyLoss(
        start_frequency=10,
        end_frequency=255,
        num_epochs=100
    )
    
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    
    loss = loss_fn(pred, target, current_epoch=10)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_charbonnier_loss():
    """Test CharbonnierLoss computation."""
    loss_fn = CharbonnierLoss(eps=1e-3)
    
    pred = torch.randn(2, 3, 64, 64)
    target = torch.randn(2, 3, 64, 64)
    
    loss = loss_fn(pred, target)
    
    assert loss.item() > 0
    assert not torch.isnan(loss)


if __name__ == '__main__':
    test_gradual_frequency_loss()
    test_charbonnier_loss()
    print("✅ All tests passed!")
