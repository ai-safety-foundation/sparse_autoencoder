"""Test the feature density metric."""

from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.feature_density import TrainBatchFeatureDensityMetric
from sparse_autoencoder.metrics.utils.find_metric_result import find_metric_result


def test_calc_feature_density() -> None:
    """Check that the feature density matches an alternative way of doing the calc."""
    activations = torch.tensor([[[0.5, 0.5, 0.0]], [[0.5, 0.0, 0.0001]], [[0.0, 0.1, 0.0]]])

    # Use different approach to check
    threshold = 0.01
    above_threshold = activations > threshold
    expected = above_threshold.sum(dim=0, dtype=torch.float) / above_threshold.shape[0]

    res = TrainBatchFeatureDensityMetric(0.001).feature_density(activations)
    assert torch.allclose(res, expected), "Output does not match the expected result."


def test_wandb_feature_density_histogram() -> None:
    """Check the Weights & Biases Histogram is created correctly."""
    feature_density = torch.tensor([[0.001, 0.001, 0.001, 0.5, 0.5, 1.0]])
    res = TrainBatchFeatureDensityMetric().wandb_feature_density_histogram(feature_density)

    # Check 0.001 is in the first bin 3 times
    expected_first_bin_value = 3
    assert res[0].histogram[0] == expected_first_bin_value


def test_calculate_aggregates() -> None:
    """Check that the metrics are aggregated in the calculate method."""
    activations = torch.tensor([[[0.5, 0.5, 0.0]], [[0.5, 0.0, 0.0001]], [[0.0, 0.1, 0.0]]])
    res = TrainBatchFeatureDensityMetric().calculate(
        TrainMetricData(
            input_activations=activations,
            learned_activations=activations,
            decoded_activations=activations,
        )
    )

    find_metric_result(res, name="feature_density")


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
    metric = TrainBatchFeatureDensityMetric()
    results = metric.calculate(data)
    weights_biases_logs = [result.wandb_log for result in results]

    assert len(weights_biases_logs) == 1, """Should only be one metric result."""
    assert (
        len(results[0].component_wise_values) == n_components
    ), """Should be one histogram per component."""
    assert weights_biases_logs == snapshot
