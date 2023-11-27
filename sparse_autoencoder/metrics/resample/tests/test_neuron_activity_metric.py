"""Tests for the NeuronActivityMetric class."""
import pytest
import torch

from sparse_autoencoder.metrics.resample.abstract_resample_metric import ResampleMetricData
from sparse_autoencoder.metrics.resample.neuron_activity_metric import NeuronActivityMetric


@pytest.fixture()
def sample_neuron_activity_data() -> ResampleMetricData:
    """Fixture to provide sample neuron activity data for testing.

    Returns:
        ResampleMetricData: A sample ResampleMetricData object with mock neuron activity.
    """
    # Create mock data with exactly 10 neurons, of which 2 are dead
    neuron_activity = torch.randint(1, 10000, (10,))
    neuron_activity[2] = 0
    neuron_activity[7] = 0
    return ResampleMetricData(neuron_activity=neuron_activity)


def test_dead_neuron_count(sample_neuron_activity_data: ResampleMetricData) -> None:
    """Test if dead neuron count is correctly calculated.

    Args:
        sample_neuron_activity_data: The sample neuron activity data for testing.
    """
    neuron_activity_metric = NeuronActivityMetric()
    metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
    expected_dead_neuron_count = 2
    assert metrics["resample_dead_neuron_count"] == expected_dead_neuron_count


def test_alive_neuron_count(sample_neuron_activity_data: ResampleMetricData) -> None:
    """Test if alive neuron count is correctly calculated.

    Args:
        sample_neuron_activity_data: The sample neuron activity data for testing.
    """
    neuron_activity_metric = NeuronActivityMetric()
    metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
    expected_alive_neuron_count = 8
    assert metrics["resample_alive_neuron_count"] == expected_alive_neuron_count


def test_histogram_generation(sample_neuron_activity_data: ResampleMetricData) -> None:
    """Test if histogram is correctly generated in the metrics.

    Args:
        sample_neuron_activity_data: The sample neuron activity data for testing.
    """
    neuron_activity_metric = NeuronActivityMetric()
    metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)

    assert "resample_neuron_activity_histogram" in metrics
