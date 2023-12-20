"""Tests for the abstract train metric class."""
import torch

from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData


def test_adds_component_dimension() -> None:
    """Test that it adds a component dimension if not initialised with one."""
    d_batch: int = 2
    n_input_output_features: int = 4
    n_learned_features: int = 8

    metric_data = TrainMetricData(
        input_activations=torch.randn(d_batch, n_input_output_features),
        learned_activations=torch.randn(d_batch, n_learned_features),
        decoded_activations=torch.randn(d_batch, n_input_output_features),
    )

    assert metric_data.input_activations.shape == (d_batch, 1, n_input_output_features)
    assert metric_data.learned_activations.shape == (d_batch, 1, n_learned_features)
    assert metric_data.decoded_activations.shape == (d_batch, 1, n_input_output_features)


def test_no_changes_with_component_dimension_already_added() -> None:
    """Test that it does not change the input if it already has a component dimension."""
    d_batch: int = 2
    n_components: int = 3
    n_input_output_features: int = 4
    n_learned_features: int = 8

    metric_data = TrainMetricData(
        input_activations=torch.randn(d_batch, n_components, n_input_output_features),
        learned_activations=torch.randn(d_batch, n_components, n_learned_features),
        decoded_activations=torch.randn(d_batch, n_components, n_input_output_features),
    )

    assert metric_data.input_activations.shape == (d_batch, n_components, n_input_output_features)
    assert metric_data.learned_activations.shape == (d_batch, n_components, n_learned_features)
    assert metric_data.decoded_activations.shape == (d_batch, n_components, n_input_output_features)
