"""Tests for the L0NormMetric class."""
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.train.l0_norm_metric import TrainBatchLearnedActivationsL0


def test_l0_norm_metric() -> None:
    """Test the L0NormMetric."""
    learned_activations = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 0.01, 2.0]]])
    l0_norm_metric = TrainBatchLearnedActivationsL0()
    data = TrainMetricData(
        input_activations=torch.zeros_like(learned_activations),
        learned_activations=learned_activations,
        decoded_activations=torch.zeros_like(learned_activations),
    )
    log = l0_norm_metric.calculate(data)
    expected = 3 / 2
    assert log[0].component_wise_values == expected


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
    metric = TrainBatchLearnedActivationsL0()
    results = metric.calculate(data)
    weights_biases_logs = [result.wandb_log for result in results]

    assert len(weights_biases_logs) == 1, """Should only be one metric result."""
    assert (
        len(results[0].component_wise_values) == n_components
    ), """Should be one histogram per component."""
    assert weights_biases_logs == snapshot
