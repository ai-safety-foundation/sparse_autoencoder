"""Tests for the capacity calculation and histogram creation."""

import math

import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.train.metrics.capacity import (
    calc_capacities, wandb_capacities_histogram)


@pytest.mark.parametrize("features,expected_capacities", [
    (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), torch.tensor([1.0, 1.0, 1.0])),
    (torch.tensor([[-0.8, -0.8, -0.8], [-0.8, -0.8, -0.8], [-0.8, -0.8, -0.8]]), torch.ones(3) / 3),
    (torch.tensor([[1.0, 0.0, 0], [math.sqrt(2), math.sqrt(2), 0.0], [0.0, 0.0, 1.0]]), torch.tensor([2 / 3, 2 / 3, 1.0])),
])
def test_calc_capacities(features: Float[Tensor, "n_feats feat_dim"], expected_capacities: Float[Tensor, " n_feats"]) -> None:
    """Check that the capacity calculation is correct."""
    capacities = calc_capacities(features)
    assert torch.allclose(capacities, expected_capacities, rtol=1e-3), "Capacity calculation is incorrect."

def test_wandb_capacity_histogram() -> None:
    """Check the Weights & Biases Histogram is created correctly."""
    capacities = torch.tensor([0.5, 0.1, 1, 1, 1])
    res = wandb_capacities_histogram(capacities)

    assert res.histogram == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
    ], "Histogram is incorrect."
