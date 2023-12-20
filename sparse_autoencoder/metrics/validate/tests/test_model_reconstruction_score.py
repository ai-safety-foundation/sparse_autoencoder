"""Test the model reconstruction score metric."""

from jaxtyping import Float
import pytest
from syrupy.session import SnapshotSession
import torch
from torch import Tensor

from sparse_autoencoder.metrics.utils.find_metric_result import find_metric_result
from sparse_autoencoder.metrics.validate.abstract_validate_metric import ValidationMetricData
from sparse_autoencoder.metrics.validate.model_reconstruction_score import ModelReconstructionScore
from sparse_autoencoder.tensor_types import Axis


def test_model_reconstruction_score_empty_data() -> None:
    """Test the model reconstruction score with empty data.

    This test validates that the method returns an empty dictionary when no data
    is provided (i.e., at the end of training or in similar scenarios).
    """
    data = ValidationMetricData(
        source_model_loss=Float[Tensor, Axis.ITEMS]([]),
        source_model_loss_with_reconstruction=Float[Tensor, Axis.ITEMS]([]),
        source_model_loss_with_zero_ablation=Float[Tensor, Axis.ITEMS]([]),
    )
    metric = ModelReconstructionScore()
    result = metric.calculate(data)
    assert result == []


@pytest.mark.parametrize(
    ("data", "expected_score"),
    [
        (
            ValidationMetricData(
                source_model_loss=torch.tensor([[3.0], [3.0], [3.0]]),
                source_model_loss_with_reconstruction=torch.tensor([[3.0], [3.0], [3.0]]),
                source_model_loss_with_zero_ablation=torch.tensor([[4.0], [4.0], [4.0]]),
            ),
            1.0,
        ),
        (
            ValidationMetricData(
                source_model_loss=Float[Tensor, Axis.ITEMS]([[0.5], [1.5], [2.5]]),
                source_model_loss_with_reconstruction=Float[Tensor, Axis.ITEMS](
                    [[1.5], [2.5], [3.5]]
                ),
                source_model_loss_with_zero_ablation=Float[Tensor, Axis.ITEMS](
                    [[8.0], [7.0], [4.0]]
                ),
            ),
            0.79,
        ),
    ],
)
def test_model_reconstruction_score_various_data(
    data: ValidationMetricData, expected_score: float
) -> None:
    """Test the model reconstruction score with various data inputs.

    This test uses parameterization to check the model reconstruction score
    calculation for different sets of input data.
    """
    metric = ModelReconstructionScore()
    calculated = metric.calculate(data)

    reconstruction_score = find_metric_result(calculated, name="reconstruction_score", postfix=None)

    result = reconstruction_score.component_wise_values
    assert isinstance(result, Tensor)
    assert round(result[0].item(), 2) == expected_score


def test_weights_biases_log_matches_snapshot(snapshot: SnapshotSession) -> None:
    """Test the log function for Weights & Biases."""
    n_items = 10
    n_components = 6

    # Create some data
    torch.manual_seed(0)
    data = ValidationMetricData(
        source_model_loss=torch.rand((n_items, n_components)),
        source_model_loss_with_reconstruction=torch.rand((n_items, n_components)),
        source_model_loss_with_zero_ablation=torch.rand((n_items, n_components)),
    )

    # Get the wandb log
    metric = ModelReconstructionScore()
    results = metric.calculate(data)
    weights_biases_logs = [result.wandb_log for result in results]

    for result in results:
        assert (
            len(result.component_wise_values) == n_components
        ), """Should be one histogram per component."""

    assert weights_biases_logs == snapshot
