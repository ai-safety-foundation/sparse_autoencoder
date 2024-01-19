"""Tests for the L0NormMetric class."""
import torch

from sparse_autoencoder.metrics.train.l0_norm_metric import L0LearnedActivations


def test_l0_norm_metric() -> None:
    """Test the L0NormMetric."""
    l0_norm_metric = L0LearnedActivations(["mlp_1", "mlp_2"])

    process_batch_size = 1
    n_components = 2
    n_input_features = 2

    input_activations = torch.zeros((process_batch_size, n_components, n_input_features))
    learned_activations = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.01, 2.0]]])

    res = l0_norm_metric(input_activations, learned_activations, input_activations)
    expected = 3 / 2
    assert res["l0_norm"] == expected
