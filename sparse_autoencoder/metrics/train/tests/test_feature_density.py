"""Test the feature density metric."""
import pytest
import torch

from sparse_autoencoder.metrics.train.feature_density import FeatureDensityMetric


@pytest.mark.parametrize(
    ("num_learned_features", "num_components", "learned_activations", "expected_output"),
    [
        pytest.param(
            3,
            1,
            torch.tensor([[[1.0, 1.0, 1.0]]]),
            torch.tensor([[1.0, 1.0, 1.0]]),
            id="Single component axis, all neurons active",
        ),
        pytest.param(
            3,
            None,
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([1.0, 1.0, 1.0]),
            id="No component axis, all neurons active",
        ),
        pytest.param(
            3,
            1,
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            id="Single component, no neurons active",
        ),
        pytest.param(
            3,
            2,
            torch.tensor(
                [
                    [  # Batch 1
                        [1.0, 0.0, 1.0],  # Component 1: learned features
                        [0.0, 1.0, 0.0],  # Component 2: learned features
                    ],
                    [  # Batch 2
                        [0.0, 1.0, 0.0],  # Component 1: learned features
                        [1.0, 0.0, 1.0],  # Component 2: learned features
                    ],
                ],
            ),
            torch.tensor(
                [
                    [0.5, 0.5, 0.5],  # Component 1: learned features
                    [0.5, 0.5, 0.5],  # Component 2: learned features
                ]
            ),
            id="Multiple components, mixed activity",
        ),
    ],
)
def test_feature_density_metric(
    num_learned_features: int,
    num_components: int,
    learned_activations: torch.Tensor,
    expected_output: torch.Tensor,
) -> None:
    """Test the FeatureDensityMetric for different scenarios.

    Args:
        num_learned_features: Number of learned features.
        num_components: Number of components.
        learned_activations: Learned activations tensor.
        expected_output: Expected output tensor.
    """
    metric = FeatureDensityMetric(num_learned_features, num_components)
    result = metric.forward(learned_activations)
    assert result.shape == expected_output.shape
    assert torch.allclose(result, expected_output)
