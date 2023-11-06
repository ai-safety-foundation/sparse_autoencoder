"""Test the feature density metric."""

import torch

from sparse_autoencoder.train.metrics.feature_density import (
    calc_feature_density,
    wandb_feature_density_histogram,
)


def test_calc_feature_density() -> None:
    """Check that the feature density matches an alternative way of doing the calc."""
    activations = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.0001], [0.0, 0.1, 0.0]])

    # Use different approach to check
    threshold = 0.01
    above_threshold = activations > threshold
    expected = above_threshold.sum(dim=0, dtype=torch.float64) / above_threshold.shape[0]

    res = calc_feature_density(activations)
    assert torch.allclose(res, expected), "Output does not match the expected result."


def test_wandb_feature_density_histogram() -> None:
    """Check the Weights & Biases Histogram is created correctly."""
    feature_density = torch.tensor([0.001, 0.001, 0.001, 0.5, 0.5, 1.0])
    res = wandb_feature_density_histogram(feature_density)

    # Check 0.001 is in the first bin 3 times
    expected_first_bin_value = 3
    assert res.histogram[0] == expected_first_bin_value
