"""Test the model reconstruction score metric."""

from jaxtyping import Float
import pytest
from torch import Tensor

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
    assert result == {}


@pytest.mark.parametrize(
    ("data", "expected_score"),
    [
        (
            ValidationMetricData(
                source_model_loss=Float[Tensor, Axis.ITEMS]([3.0, 3.0, 3.0]),
                source_model_loss_with_reconstruction=Float[Tensor, Axis.ITEMS]([3.0, 3.0, 3.0]),
                source_model_loss_with_zero_ablation=Float[Tensor, Axis.ITEMS]([4.0, 4.0, 4.0]),
            ),
            1.0,
        ),
        (
            ValidationMetricData(
                source_model_loss=Float[Tensor, Axis.ITEMS]([0.5, 1.5, 2.5]),
                source_model_loss_with_reconstruction=Float[Tensor, Axis.ITEMS]([1.5, 2.5, 3.5]),
                source_model_loss_with_zero_ablation=Float[Tensor, Axis.ITEMS]([8.0, 7.0, 4.0]),
            ),
            0.67,
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
    result = metric.calculate(data)
    assert round(result["validate/model_reconstruction_score"], 2) == expected_score
