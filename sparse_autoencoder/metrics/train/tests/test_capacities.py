"""Tests for the capacity calculation and histogram creation."""

import math

from jaxtyping import Float
import pytest
from syrupy.session import SnapshotSession
import torch
from torch import Tensor

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.capacity import CapacityMetric
from sparse_autoencoder.metrics.utils.find_metric_result import find_metric_result
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.parametrize(
    ("features", "expected_capacities"),
    [
        pytest.param(
            torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]]),
            torch.tensor([[1.0, 1.0]]),
            id="orthogonal",
        ),
        pytest.param(
            torch.tensor(
                [[[1.0, 0.0, 0.0], [-0.8, -0.8, -0.8]], [[0.0, 1.0, 0.0], [-0.8, -0.8, -0.8]]]
            ),
            torch.tensor([[1.0, 1.0], [0.5, 0.5]]),
            id="orthogonal_2_components",
        ),
        pytest.param(
            torch.tensor([[[-0.8, -0.8, -0.8]], [[-0.8, -0.8, -0.8]]]),
            torch.ones(2).unsqueeze(0) / 2,
            id="same_feature",
        ),
        pytest.param(
            torch.tensor(
                [[[1.0, 0.0, 0]], [[1 / math.sqrt(2), 1 / math.sqrt(2), 0.0]], [[0.0, 0.0, 1.0]]]
            ),
            torch.tensor([2 / 3, 2 / 3, 1.0]).unsqueeze(0),
        ),
    ],
)
def test_calc_capacities(
    features: Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)],
    expected_capacities: Float[Tensor, Axis.BATCH],
) -> None:
    """Check that the capacity calculation is correct."""
    capacities = CapacityMetric.capacities(features)
    assert torch.allclose(
        capacities, expected_capacities, rtol=1e-3
    ), "Capacity calculation is incorrect."


def test_calculate_returns_histogram() -> None:
    """Check the calculate function returns a histogram."""
    metric = CapacityMetric()
    activations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    res = metric.calculate(
        TrainMetricData(
            input_activations=activations,
            learned_activations=activations,
            decoded_activations=activations,
        )
    )
    find_metric_result(res, name="capacities")


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
    metric = CapacityMetric()
    results = metric.calculate(data)
    weights_biases_logs = [result.wandb_log for result in results]

    assert len(weights_biases_logs) == 1, """Should only be one metric result."""
    assert (
        len(results[0].component_wise_values) == n_components
    ), """Should be one histogram per component."""
    assert weights_biases_logs == snapshot
