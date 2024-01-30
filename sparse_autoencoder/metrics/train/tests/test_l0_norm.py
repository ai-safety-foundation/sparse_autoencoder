"""Test the L0 norm sparsity metric."""
import pytest
import torch

from sparse_autoencoder.metrics.train.l0_norm import (
    L0NormMetric,  # Adjust the import path as needed
)


@pytest.mark.parametrize(
    ("num_components", "learned_activations", "expected_output"),
    [
        pytest.param(
            1,
            torch.tensor([[[1.0, 0.0, 1.0]]]),
            torch.tensor(2.0),
            id="Single component, mixed activity",
        ),
        pytest.param(
            None,
            torch.tensor([[1.0, 0.0, 1.0]]),
            torch.tensor(2.0),
            id="No component axis, mixed activity",
        ),
        pytest.param(
            1,
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.tensor(0.0),
            id="Single component, no neurons active",
        ),
        pytest.param(
            2,
            torch.tensor(
                [
                    [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                ]
            ),
            torch.tensor([1.5, 2.0]),
            id="Multiple components, mixed activity",
        ),
    ],
)
def test_l0_norm_metric(
    num_components: int,
    learned_activations: torch.Tensor,
    expected_output: torch.Tensor,
) -> None:
    """Test the L0NormMetric for different scenarios.

    Args:
        num_components: Number of components.
        learned_activations: Learned activations tensor.
        expected_output: Expected output tensor.
    """
    metric = L0NormMetric(num_components)
    result = metric.forward(learned_activations)

    assert result.shape == expected_output.shape
    assert torch.allclose(result, expected_output)
