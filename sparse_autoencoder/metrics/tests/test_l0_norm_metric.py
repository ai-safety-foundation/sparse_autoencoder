"""Tests for the L0NormMetric class."""
import pytest
import torch

from sparse_autoencoder.metrics.l0_norm_metric import L0NormMetric
from sparse_autoencoder.metrics.abstract_metric import TrainMetricData

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
    log = l0_norm_metric.create_weights_and_biases_log(data)
    assert log["l0_norm"] == 1.5

