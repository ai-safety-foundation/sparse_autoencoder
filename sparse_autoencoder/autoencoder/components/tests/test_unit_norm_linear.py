"""Tests for the constrained unit norm linear layer."""
import torch

from sparse_autoencoder.autoencoder.components.unit_norm_linear import ConstrainedUnitNormLinear


def test_initialization() -> None:
    """Test that the weights are initialized with unit norm."""
    layer = ConstrainedUnitNormLinear(learnt_features=3, decoded_features=4)
    weight_norms = torch.norm(layer.weight, dim=1)
    assert torch.allclose(weight_norms, torch.ones_like(weight_norms))


def test_forward_pass() -> None:
    """Test the forward pass of the layer."""
    layer = ConstrainedUnitNormLinear(learnt_features=3, decoded_features=4)
    input_tensor = torch.randn(5, 3)  # Batch size of 5, learnt features of 3
    output = layer(input_tensor)
    assert output.shape == (5, 4)  # Batch size of 5, decoded features of 4


def test_bias_initialization_and_usage() -> None:
    """Test the bias is initialized and used correctly."""
    layer = ConstrainedUnitNormLinear(learnt_features=3, decoded_features=4, bias=True)
    assert layer.bias is not None
    # Check the bias is used in the forward pass
    input_tensor = torch.zeros(5, 3)
    output = layer(input_tensor)
    assert torch.allclose(output, layer.bias.unsqueeze(0).expand(5, -1))


def test_multiple_training_steps() -> None:
    """Test the unit norm constraint over multiple training iterations."""
    torch.random.manual_seed(0)
    layer = ConstrainedUnitNormLinear(learnt_features=3, decoded_features=4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(10):
        data = torch.randn(5, 3)
        optimizer.zero_grad()
        logits = layer(data)

        weight_norms = torch.norm(layer.weight, dim=1)
        assert torch.allclose(weight_norms, torch.ones_like(weight_norms), atol=2e-3)

        loss = torch.mean(logits**2)
        loss.backward()
        optimizer.step()


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
