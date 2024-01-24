"""Test the lightning-wrapped SAE."""

from sparse_autoencoder.autoencoder.lightning import LitSparseAutoencoder
from sparse_autoencoder.autoencoder.model import SparseAutoencoderConfig


def test_initialises() -> None:
    """Check it can be initialised."""
    LitSparseAutoencoder(
        SparseAutoencoderConfig(n_components=3, n_input_features=4, n_learned_features=8),
        ["a", "b", "c"],
    )
