"""Tests for the L0NormMetric class."""
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.l0_norm_metric import TrainBatchLearnedActivationsL0


def test_l0_norm_metric() -> None:
    """Test the L0NormMetric."""
    learned_activations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.01, 2.0]])
    l0_norm_metric = TrainBatchLearnedActivationsL0()
    data = TrainMetricData(
        input_activations=torch.zeros_like(learned_activations),
        learned_activations=learned_activations,
        decoded_activations=torch.zeros_like(learned_activations),
    )
    log = l0_norm_metric.calculate(data)
    expected = 3 / 2
    assert log["learned_activations_l0_norm"] == expected
