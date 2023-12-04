"""Test the feature density metric."""

import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.feature_density import TrainBatchFeatureDensityMetric


def test_calc_feature_density() -> None:
    """Check that the feature density matches an alternative way of doing the calc."""
    activations = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.0001], [0.0, 0.1, 0.0]])

    # Use different approach to check
    threshold = 0.01
    above_threshold = activations > threshold
    expected = above_threshold.sum(dim=0, dtype=torch.float) / above_threshold.shape[0]

    res = TrainBatchFeatureDensityMetric(0.001).feature_density(activations)
    assert torch.allclose(res, expected), "Output does not match the expected result."


def test_wandb_feature_density_histogram() -> None:
    """Check the Weights & Biases Histogram is created correctly."""
    feature_density = torch.tensor([0.001, 0.001, 0.001, 0.5, 0.5, 1.0])
    res = TrainBatchFeatureDensityMetric().wandb_feature_density_histogram(feature_density)

    # Check 0.001 is in the first bin 3 times
    expected_first_bin_value = 3
    assert res.histogram[0] == expected_first_bin_value


def test_calculate_aggregates() -> None:
    """Check that the metrics are aggregated in the calculate method."""
    activations = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.0001], [0.0, 0.1, 0.0]])
    res = TrainBatchFeatureDensityMetric().calculate(
        TrainMetricData(
            input_activations=activations,
            learned_activations=activations,
            decoded_activations=activations,
        )
    )

    # Check both metrics are in the result
    assert "train/batch_feature_density_histogram" in res
