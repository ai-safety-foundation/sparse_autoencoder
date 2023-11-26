"""Test the model reconstruction score metric."""

import pytest

from sparse_autoencoder.metrics.validate.abstract_validate_metric import ValidationMetricData
from sparse_autoencoder.metrics.validate.model_reconstruction_score import ModelReconstructionScore
from sparse_autoencoder.tensor_types import ValidationStatistics


def test_model_reconstruction_score_empty_data() -> None:
    """Test the model reconstruction score with empty data.

    This test validates that the method returns an empty dictionary when no data
    is provided (i.e., at the end of training or in similar scenarios).
    """
    data = ValidationMetricData(
        source_model_loss=ValidationStatistics([]),
        source_model_loss_with_reconstruction=ValidationStatistics([]),
        source_model_loss_with_zero_ablation=ValidationStatistics([]),
    )
    metric = ModelReconstructionScore()
    result = metric.calculate(data)
    assert result == {}


@pytest.mark.parametrize(
    ("data", "expected_score"),
    [
        (
            ValidationMetricData(
                source_model_loss=ValidationStatistics([3.0, 3.0, 3.0]),
                source_model_loss_with_reconstruction=ValidationStatistics([3.0, 3.0, 3.0]),
                source_model_loss_with_zero_ablation=ValidationStatistics([4.0, 4.0, 4.0]),
            ),
            1.0,
        ),
        (
            ValidationMetricData(
                source_model_loss=ValidationStatistics([0.5, 1.5, 2.5]),
                source_model_loss_with_reconstruction=ValidationStatistics([1.5, 2.5, 3.5]),
                source_model_loss_with_zero_ablation=ValidationStatistics([8.0, 7.0, 4.0]),
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
    assert round(result["model_reconstruction_score"], 2) == expected_score
