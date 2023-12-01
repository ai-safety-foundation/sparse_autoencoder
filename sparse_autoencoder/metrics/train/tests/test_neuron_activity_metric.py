"""Tests for the NeuronActivityMetric class."""
import pytest
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.neuron_activity_metric import NeuronActivityMetric


@pytest.fixture()
def sample_neuron_activity_data() -> TrainMetricData:
    """Fixture to provide sample neuron activity data for testing.

    Returns:
        TrainMetricData: A sample TrainMetricData object with mock neuron activity.
    """
    # Create mock data with exactly 10 neurons, of which 2 are inactive
    activations = torch.rand((1, 10))  # Mock activation for 10 neurons
    activations[:, 2] = 0  # Mock dead neuron
    activations[:, 7] = 0  # Mock dead neuron
    return TrainMetricData(
        learned_activations=activations,
        input_activations=torch.zeros((1, 10)),
        decoded_activations=torch.zeros((1, 10)),
    )


def test_dead_neuron_count(sample_neuron_activity_data: TrainMetricData) -> None:
    """Test if dead neuron count is correctly calculated.

    Args:
        sample_neuron_activity_data: The sample neuron activity data for testing.
    """
    neuron_activity_metric = NeuronActivityMetric(horizon=1)
    metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
    expected_dead_neuron_count = 2
    assert metrics["dead_neuron_count"] == expected_dead_neuron_count


def test_alive_neuron_count(sample_neuron_activity_data: TrainMetricData) -> None:
    """Test if alive neuron count is correctly calculated.

    Args:
        sample_neuron_activity_data: The sample neuron activity data for testing.
    """
    neuron_activity_metric = NeuronActivityMetric(horizon=1)
    metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
    expected_alive_neuron_count = 8
    assert metrics["alive_neuron_count"] == expected_alive_neuron_count


def test_histogram_generation(sample_neuron_activity_data: TrainMetricData) -> None:
    """Test if histogram is correctly generated in the metrics.

    Args:
        sample_neuron_activity_data: The sample neuron activity data for testing.
    """
    neuron_activity_metric = NeuronActivityMetric(horizon=5)
    for _ in range(4):
        metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
        assert metrics == {}

    metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
    assert "neuron_activity_histogram" in metrics
    assert "log_neuron_activity_histogram" in metrics
