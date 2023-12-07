"""Tests for the NeuronActivityMetric class."""
import pytest
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.neuron_activity_metric import (
    NeuronActivityHorizonData,
    NeuronActivityMetric,
)


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


class TestNeuronActivityHorizonData:
    """Test the NeuronActivityHorizonData class."""

    def test_initialisation(self) -> None:
        """Test it initialises without errors."""
        NeuronActivityHorizonData(
            approximate_activation_horizon=5,
            train_batch_size=2,
            number_learned_features=10,
            thresholds=[0.5],
        )

    def test_step_calculates_when_at_horizon(
        self,
    ) -> None:
        """Test that step triggers a calculation when expected."""
        horizon_in_steps = 2
        train_batch_size = 2

        threshold_data_store = NeuronActivityHorizonData(
            approximate_activation_horizon=int(horizon_in_steps * train_batch_size),
            train_batch_size=train_batch_size,
            number_learned_features=4,
            thresholds=[0.5],
        )

        for step in range(1, 10):
            data = torch.randint(0, 2, (1, 4)).squeeze()
            res = threshold_data_store.step(data)

            if step % horizon_in_steps == 0:
                assert len(res.keys()) > 0
            else:
                assert len(res.keys()) == 0

    def test_results(self) -> None:
        """Test that the results are calculated correctly."""
        threshold_data_store = NeuronActivityHorizonData(
            approximate_activation_horizon=30,
            train_batch_size=30,
            number_learned_features=5,
            thresholds=[0.5],
        )

        data = torch.tensor([0, 30, 4, 1, 0])
        res = threshold_data_store.step(data)

        expected_dead = 2
        expected_alive = 3
        expected_almost_dead = 4

        assert res["train/activity/over_30_activations/dead_count"] == expected_dead
        assert res["train/activity/over_30_activations/alive_count"] == expected_alive
        assert res["train/activity/over_30_activations/almost_dead_0.5"] == expected_almost_dead


class TestNeuronActivityMetric:
    """Test the NeuronActivityMetric class."""

    def test_dead_neuron_count(self, sample_neuron_activity_data: TrainMetricData) -> None:
        """Test if dead neuron count is correctly calculated.

        Args:
            sample_neuron_activity_data: The sample neuron activity data for testing.
        """
        neuron_activity_metric = NeuronActivityMetric(approximate_horizons=[1])
        metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
        expected_dead_neuron_count = 2
        assert metrics["train/activity/over_1_activations/dead_count"] == expected_dead_neuron_count

    def test_alive_neuron_count(self, sample_neuron_activity_data: TrainMetricData) -> None:
        """Test if alive neuron count is correctly calculated.

        Args:
            sample_neuron_activity_data: The sample neuron activity data for testing.
        """
        neuron_activity_metric = NeuronActivityMetric(approximate_horizons=[1])
        metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
        expected_alive_neuron_count = 8
        assert (
            metrics["train/activity/over_1_activations/alive_count"] == expected_alive_neuron_count
        )

    def test_histogram_generation(self, sample_neuron_activity_data: TrainMetricData) -> None:
        """Test if histogram is correctly generated in the metrics.

        Args:
            sample_neuron_activity_data: The sample neuron activity data for testing.
        """
        neuron_activity_metric = NeuronActivityMetric(approximate_horizons=[5])
        for _ in range(4):
            metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
            assert metrics == {}

        metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
        assert "train/activity/over_5_activations/activity_histogram" in metrics
        assert "train/activity/over_5_activations/log_activity_histogram" in metrics
