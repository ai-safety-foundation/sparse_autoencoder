"""Tests for the constrained unit norm linear layer."""
import torch

from sparse_autoencoder.autoencoder.components.unit_norm_decoder import UnitNormDecoder


def test_initialization() -> None:
    """Test that the weights are initialized with unit norm."""
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4)
    weight_norms = torch.norm(layer.weight, dim=1)
    assert torch.allclose(weight_norms, torch.ones_like(weight_norms))


def test_forward_pass() -> None:
    """Test the forward pass of the layer."""
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4)
    input_tensor = torch.randn(5, 3)  # Batch size of 5, learnt features of 3
    output = layer(input_tensor)
    assert output.shape == (5, 4)  # Batch size of 5, decoded features of 4


def test_multiple_training_steps() -> None:
    """Test the unit norm constraint over multiple training iterations."""
    torch.random.manual_seed(0)
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(4):
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
    torch.random.manual_seed(42)
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0)
    data = torch.randn((1, 3), requires_grad=True)
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
    assert torch.allclose(weight_norms, torch.ones_like(weight_norms), atol=0.02, rtol=False)


def test_unit_norm_decreases() -> None:
    """Check that the unit norm is applied after each gradient step."""
    for _ in range(4):
        data = torch.randn((1, 3), requires_grad=True)

        # run with the backward hook
        layer = UnitNormDecoder(learnt_features=3, decoded_features=4)
        layer_weights = torch.nn.Parameter(layer.weight.clone())
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0)
        logits = layer(data)
        loss = torch.mean(logits**2)
        loss.backward()
        optimizer.step()
        weight_norms_with_hook = torch.sum(layer.weight**2, dim=1).clone()

        # Run without the hook
        layer_without_hook = UnitNormDecoder(
            learnt_features=3, decoded_features=4, enable_gradient_hook=False
        )
        layer_without_hook._weight = layer_weights  # type: ignore (testing only)  # noqa: SLF001
        optimizer_without_hook = torch.optim.SGD(
            layer_without_hook.parameters(), lr=0.1, momentum=0
        )
        logits_without_hook = layer_without_hook(data)
        loss_without_hook = torch.mean(logits_without_hook**2)
        loss_without_hook.backward()
        optimizer_without_hook.step()
        weight_norms_without_hook = torch.sum(layer_without_hook.weight**2, dim=1).clone()

        # Check that the norm with the hook is closer to 1 than without the hook
        target_norms = torch.ones_like(weight_norms_with_hook)
        absolute_diff_with_hook = torch.abs(weight_norms_with_hook - target_norms)
        absolute_diff_without_hook = torch.abs(weight_norms_without_hook - target_norms)
        assert torch.all(absolute_diff_with_hook < absolute_diff_without_hook)
