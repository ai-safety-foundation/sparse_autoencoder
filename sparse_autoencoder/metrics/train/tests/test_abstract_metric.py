"""Tests for the AbstractMetric class."""
from typing import final

import pytest
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


@final
class DummyMetric(AbstractTrainMetric):
    """Dummy metric for testing.

    Returns a sum of the learned activations.
    """

    def calculate(self, data: TrainMetricData) -> dict[str, float]:
        """Create a log item for Weights and Biases."""
        return {"dummy_metric": data.learned_activations.sum().item()}


@pytest.fixture()
def dummy_metric() -> DummyMetric:
    """Dummy metric for testing."""
    return DummyMetric()


def test_abstract_class_enforced() -> None:
    """Test that initializing the abstract class raises an error."""
    with pytest.raises(TypeError):
        AbstractTrainMetric()  # type: ignore


def test_create_weights_and_biases_log(dummy_metric: DummyMetric) -> None:
    """Test the calculate method."""
    data = TrainMetricData(
        input_activations=torch.ones((1, 3)),
        learned_activations=torch.ones((1, 3)),
        decoded_activations=torch.ones((1, 3)),
    )
    log = dummy_metric.calculate(data)
    expected = 3.0
    assert log["dummy_metric"] == expected
