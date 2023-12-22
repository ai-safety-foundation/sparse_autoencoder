"""Tests for the NeuronActivityMetric class."""
from jaxtyping import Float, Int64
import pytest
from syrupy.session import SnapshotSession
import torch
from torch import Tensor

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.neuron_activity_metric import (
    NeuronActivityHorizonData,
    NeuronActivityMetric,
)
from sparse_autoencoder.metrics.utils.find_metric_result import find_metric_result
from sparse_autoencoder.tensor_types import Axis


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

    @pytest.mark.parametrize(
        ("n_components"),
        [
            pytest.param(1, id="1 component"),
            pytest.param(2, id="2 components"),
        ],
    )
    def test_initialisation(self, n_components: int) -> None:
        """Test it initialises without errors."""
        NeuronActivityHorizonData(
            approximate_activation_horizon=5,
            train_batch_size=2,
            n_learned_features=10,
            thresholds=[0.5],
            n_components=n_components,
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
            n_learned_features=4,
            thresholds=[0.5],
            n_components=1,
        )

        for step in range(1, 10):
            data = torch.randint(0, 2, (1, 4))
            res = threshold_data_store.step(data)

            if step % horizon_in_steps == 0:
                assert len(res) > 0
            else:
                assert len(res) == 0

    def test_results_match_expectations(self) -> None:
        """Test that the results are calculated correctly."""
        threshold_data_store = NeuronActivityHorizonData(
            approximate_activation_horizon=30,
            train_batch_size=30,
            n_learned_features=5,
            thresholds=[0.5],
            n_components=1,
        )

        data = torch.tensor([[0, 30, 4, 1, 0]])
        res = threshold_data_store.step(data)

        expected_dead = 2
        expected_alive = 3
        expected_almost_dead = 4

        dead_over_30_activations = find_metric_result(res, postfix="dead_over_30_activations")
        assert dead_over_30_activations.component_wise_values[0] == expected_dead

        alive_over_30_activations = find_metric_result(res, postfix="alive_over_30_activations")
        assert alive_over_30_activations.component_wise_values[0] == expected_alive

        almost_dead_over_30_activations = find_metric_result(
            res, postfix="almost_dead_5.0e-01_over_30_activations"
        )
        assert almost_dead_over_30_activations.component_wise_values[0] == expected_almost_dead


class TestNeuronActivityMetric:
    """Test the NeuronActivityMetric class."""

    @pytest.mark.parametrize(
        ("learned_activations", "expected_dead_count", "expected_alive_count"),
        [
            pytest.param(
                torch.tensor([[0.0, 0, 0, 0, 0]]),
                torch.tensor([5]),
                torch.tensor([0]),
                id="All dead",
            ),
            pytest.param(
                torch.tensor([[1.0, 1, 1, 1, 1]]),
                torch.tensor([0]),
                torch.tensor([5]),
                id="All alive",
            ),
            pytest.param(
                torch.tensor([[0.0, 1, 0, 1, 0]]),
                torch.tensor([3]),
                torch.tensor([2]),
                id="Some dead",
            ),
            pytest.param(
                torch.tensor([[[0.0, 1, 0, 1, 0], [0.0, 0, 0, 0, 0]]]),
                torch.tensor([3, 5]),
                torch.tensor([2, 0]),
                id="Multiple components with some dead",
            ),
        ],
    )
    def test_dead_neuron_count(
        self,
        learned_activations: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)],
        expected_dead_count: Int64[Tensor, Axis.names(Axis.COMPONENT)],
        expected_alive_count: Int64[Tensor, Axis.names(Axis.COMPONENT)],
    ) -> None:
        """Test if dead neuron count is correctly calculated."""
        input_activations = torch.zeros_like(learned_activations, dtype=torch.float)
        data = TrainMetricData(
            learned_activations=learned_activations,
            # Input and decoded activations are not used in this metric
            input_activations=input_activations,
            decoded_activations=input_activations,
        )
        neuron_activity_metric = NeuronActivityMetric(approximate_horizons=[1])
        metrics = neuron_activity_metric.calculate(data)

        dead_over_1_activations = find_metric_result(metrics, postfix="dead_over_1_activations")
        alive_over_1_activations = find_metric_result(metrics, postfix="alive_over_1_activations")

        assert isinstance(dead_over_1_activations.component_wise_values, torch.Tensor)
        assert isinstance(alive_over_1_activations.component_wise_values, torch.Tensor)
        assert torch.allclose(dead_over_1_activations.component_wise_values, expected_dead_count)
        assert torch.allclose(alive_over_1_activations.component_wise_values, expected_alive_count)

    def test_alive_neuron_count(self, sample_neuron_activity_data: TrainMetricData) -> None:
        """Test if alive neuron count is correctly calculated.

        Args:
            sample_neuron_activity_data: The sample neuron activity data for testing.
        """
        neuron_activity_metric = NeuronActivityMetric(approximate_horizons=[1])
        metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)
        expected_alive_neuron_count = 8
        alive_over_1_activations = find_metric_result(metrics, postfix="alive_over_1_activations")
        assert alive_over_1_activations.component_wise_values[0] == expected_alive_neuron_count

    def test_histogram_generation(self, sample_neuron_activity_data: TrainMetricData) -> None:
        """Test if histogram is correctly generated in the metrics.

        Args:
            sample_neuron_activity_data: The sample neuron activity data for testing.
        """
        neuron_activity_metric = NeuronActivityMetric(approximate_horizons=[5])
        for _ in range(4):
            metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)

        metrics = neuron_activity_metric.calculate(sample_neuron_activity_data)

        find_metric_result(metrics, postfix="activity_histogram_over_5_activations")
        find_metric_result(metrics, postfix="log_activity_histogram_over_5_activations")


def test_weights_biases_log_matches_snapshot(snapshot: SnapshotSession) -> None:
    """Test the log function for Weights & Biases."""
    n_batches = 10
    n_components = 6
    n_input_features = 4
    n_learned_features = 8

    # Create some data
    torch.manual_seed(0)
    data = TrainMetricData(
        input_activations=torch.rand((n_batches, n_components, n_input_features)),
        learned_activations=torch.rand((n_batches, n_components, n_learned_features)),
        decoded_activations=torch.rand((n_batches, n_components, n_input_features)),
    )

    # Get the wandb log
    metric = NeuronActivityMetric(approximate_horizons=[n_batches])
    results = metric.calculate(data)
    weights_biases_logs = [result.wandb_log for result in results]

    assert (
        len(results[0].component_wise_values) == n_components
    ), """Should be one histogram per component."""
    assert weights_biases_logs == snapshot
