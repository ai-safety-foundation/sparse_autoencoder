"""Tests for the constrained unit norm linear layer."""
import torch

from sparse_autoencoder.autoencoder.components.unit_norm_linear import ConstrainedUnitNormLinear


def test_unit_norm_applied_backward() -> None:
    """Check that the unit norm is applied after each gradient step."""
    torch.random.manual_seed(0)
    layer = ConstrainedUnitNormLinear(learnt_features=3, decoded_features=4)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0)
    data = torch.randn((3), requires_grad=True)
    logits = layer(data)
    loss = torch.mean(logits**2)
    loss.backward()

    # Check that the gradient is not zero (as that would be a trivial way the weights could be kept
    # unit norm)
    grad = layer.weight.grad
    assert grad is not None
    assert not torch.allclose(grad, torch.zeros_like(grad))

    optimizer.step()

    # Check that the weights still have unit norm
    weight_norms = torch.sum(layer.weight**2, dim=1)
    assert torch.allclose(weight_norms, torch.ones_like(weight_norms), atol=2e-3)
