"""Tests for the L0NormMetric class."""
import pytest
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.l0_norm_metric import L0NormMetric


@pytest.fixture()
def l0_norm_metric() -> L0NormMetric:
    """Fixture for L0NormMetric."""
    return L0NormMetric()


def test_l0_norm_metric(l0_norm_metric: L0NormMetric) -> None:
    """Test the L0NormMetric."""
    learned_activations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.01, 2.0]])
    data = TrainMetricData(
        input_activations=torch.zeros_like(learned_activations),
        learned_activations=learned_activations,
        decoded_activations=torch.zeros_like(learned_activations),
    )
    log = l0_norm_metric.calculate(data)
    expected = 2 / 3
    assert log["l0_norm"] == expected
