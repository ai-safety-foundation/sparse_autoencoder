"""Test the Neuron Fired Count Metric."""
import pytest
import torch

from sparse_autoencoder.metrics.train.neuron_fired_count import NeuronFiredCountMetric


@pytest.mark.parametrize(
    (
        "num_learned_features",
        "num_components",
        "threshold_is_dead_portion_fires",
        "learned_activations",
        "expected_output",
    ),
    [
        pytest.param(
            3,
            1,
            0,
            torch.tensor(
                [
                    [  # Batch 1
                        [1.0, 0.0, 1.0]  # Component 1: learned features (2 active neurons)
                    ],
                    [  # Batch 2
                        [0.0, 0.0, 0.0]  # Component 1: learned features (0 active neuron)
                    ],
                ]
            ),
            torch.tensor([[1, 0, 1]]),
            id="Single component, one dead neuron",
        ),
        pytest.param(
            3,
            None,
            0,
            torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            torch.tensor([1, 0, 1]),
            id="No component axis, one dead neuron",
        ),
        pytest.param(
            3,
            1,
            0.0,
            torch.tensor([[[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]]),
            torch.tensor([[1, 1, 1]]),
            id="Single component, no dead neurons",
        ),
        pytest.param(
            3,
            2,
            0,
            torch.tensor(
                [
                    [  # Batch 1
                        [1.0, 0.0, 1.0],  # Component 1: learned features
                        [1.0, 0.0, 1.0],  # Component 2: learned features
                    ],
                    [  # Batch 2
                        [0.0, 1.0, 0.0],  # Component 1: learned features
                        [1.0, 0.0, 1.0],  # Component 2: learned features
                    ],
                ]
            ),
            torch.tensor([[1, 1, 1], [2, 0, 2]]),
            id="Multiple components, mixed dead neurons",
        ),
    ],
)
def test_neuron_fired_count_metric(
    num_learned_features: int,
    num_components: int,
    threshold_is_dead_portion_fires: float,
    learned_activations: torch.Tensor,
    expected_output: torch.Tensor,
) -> None:
    """Test the NeuronFiredCount for different scenarios.

    Args:
        num_learned_features: Number of learned features.
        num_components: Number of components.
        threshold_is_dead_portion_fires: Threshold for counting a neuron as dead.
        learned_activations: Learned activations tensor.
        expected_output: Expected number of dead neurons.
    """
    metric = NeuronFiredCountMetric(num_learned_features, num_components)
    result = metric.forward(learned_activations)
    assert result.shape == expected_output.shape
    assert torch.equal(result, expected_output)
