"""Test the lightning-wrapped SAE."""

import torch

from sparse_autoencoder.autoencoder.lightning import LitSparseAutoencoder
from sparse_autoencoder.autoencoder.model import SparseAutoencoderConfig


def test_initialises() -> None:
    """Check it can be initialised."""
    LitSparseAutoencoder(
        SparseAutoencoderConfig(n_components=3, n_input_features=4, n_learned_features=8),
        ["a", "b", "c"],
    )


def test_forward() -> None:
    """Test the forward pass works with a simple example."""
    n_components = 5
    n_input_features = 3
    inputs = torch.randn((10, n_components, n_input_features))
    model = LitSparseAutoencoder(
        SparseAutoencoderConfig(
            n_components=n_components, n_input_features=n_input_features, n_learned_features=8
        ),
        ["a", "b", "c"],
    )
    model.forward(inputs)


def test_training_step_loss_shape() -> None:
    """Test the training step returns the correct loss shape."""
    n_components = 5
    n_input_features = 3
    n_learned_features = 8
    config = SparseAutoencoderConfig(
        n_components=n_components,
        n_input_features=n_input_features,
        n_learned_features=n_learned_features,
    )
    component_names = ["a", "b", "c"]
    model = LitSparseAutoencoder(config, component_names)
    batch = torch.randn((12, n_components, n_input_features))
    loss = model.training_step(batch, 0)
    assert loss.shape == ()
