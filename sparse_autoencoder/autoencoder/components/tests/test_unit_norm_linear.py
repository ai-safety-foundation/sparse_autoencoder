"""Tests for the constrained unit norm linear layer."""
import torch

from sparse_autoencoder.autoencoder.components.unit_norm_linear import ConstrainedUnitNormLinear


def test_unit_norm_applied_backward() -> None:
    """Check that the unit norm is applied after each gradient step."""
    layer = ConstrainedUnitNormLinear(3, 4)
    data = torch.randn((3), requires_grad=True)
    logits = layer(data)
    loss = torch.sum(logits**2)
    loss.backward()
    weight_norms = torch.sum(layer.weight**2, dim=1)

    # Check that the weights still have unit norm
    assert torch.allclose(weight_norms, torch.ones_like(weight_norms))

    # Check that the gradient is not zero (as that would be a trivial way the weights could be kept
    # unit norm)
    grad = layer.weight.grad
    assert grad is not None
    assert not torch.allclose(grad, torch.zeros_like(grad))
