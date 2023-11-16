"""Tests for the capacity calculation and histogram creation."""

import math

import pytest
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.tensor_types import LearnedActivationBatch, TrainBatchStatistic
from sparse_autoencoder.train.metrics.capacity import calc_capacities, wandb_capacities_histogram


@pytest.mark.parametrize(
    ("features", "expected_capacities"),
    [
        (
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            torch.tensor([1.0, 1.0]),
        ),
        (
            torch.tensor([[-0.8, -0.8, -0.8], [-0.8, -0.8, -0.8]]),
            torch.ones(2) / 2,
        ),
        (
            torch.tensor(
                [[1.0, 0.0, 0], [1 / math.sqrt(2), 1 / math.sqrt(2), 0.0], [0.0, 0.0, 1.0]]
            ),
            torch.tensor([2 / 3, 2 / 3, 1.0]),
        ),
    ],
)
def test_calc_capacities(
    features: LearnedActivationBatch, expected_capacities: TrainBatchStatistic
) -> None:
    """Check that the capacity calculation is correct."""
    capacities = calc_capacities(features)
    assert torch.allclose(
        capacities, expected_capacities, rtol=1e-3
    ), "Capacity calculation is incorrect."


def test_wandb_capacity_histogram(snapshot: SnapshotSession) -> None:
    """Check the Weights & Biases Histogram is created correctly."""
    capacities = torch.tensor([0.5, 0.1, 1, 1, 1])
    res = wandb_capacities_histogram(capacities)

    assert res.histogram == snapshot
