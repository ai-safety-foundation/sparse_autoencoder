"""Sparse Autoencoder Model Tests."""
import pytest
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.autoencoder.utils.tensor_shape import shape_with_optional_dimensions


def test_initialize_tied_bias() -> None:
    """Check the tied bias is initialised correctly."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)
    assert torch.allclose(model.tied_bias, geometric_median)


def test_encoded_decoded_shape_same() -> None:
    """Check the input and output are the same shape."""
    model = SparseAutoencoder(3, 6)
    input_tensor = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    output = model(input_tensor)

    assert output[1].shape == input_tensor.shape


def test_can_get_encoder_weights() -> None:
    """Check we can access the encoder weights."""
    model = SparseAutoencoder(3, 6)
    encoder = model.encoder
    assert encoder.weight.shape == (6, 3)


def test_representation(snapshot: SnapshotSession) -> None:
    """Check the string representation of the model."""
    model = SparseAutoencoder(3, 6)
    assert snapshot == str(model), "Model string representation has changed."


@pytest.mark.parametrize("n_components", [None, 1, 3])
def test_forward_pass_works(n_components: int | None) -> None:
    """Check the forward pass works without errors."""
    batch_size = 3
    n_input_features = 6
    n_learned_features = 12

    model = SparseAutoencoder(n_input_features, n_learned_features, n_components=n_components)

    input_tensor = torch.randn(
        shape_with_optional_dimensions(batch_size, n_components, n_input_features)
    )

    output = model.forward(input_tensor)
    assert output[1].shape == input_tensor.shape
