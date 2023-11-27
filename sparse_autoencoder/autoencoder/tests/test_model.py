"""Sparse Autoencoder Model Tests."""
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.autoencoder.model import SparseAutoencoder


def test_initialize_tied_bias() -> None:
    """Check the tied bias is initialised correctly."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)
    assert torch.allclose(model.tied_bias, geometric_median)


def test_encoded_decoded_shape_same() -> None:
    """Check the input and output are the same shape."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)
    input_tensor = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    output = model(input_tensor)

    assert output[1].shape == input_tensor.shape


def test_can_get_encoder_weights() -> None:
    """Check we can access the encoder weights."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)
    encoder = model.encoder
    assert encoder.weight.shape == (6, 3)


def test_representation(snapshot: SnapshotSession) -> None:
    """Check the string representation of the model."""
    model = SparseAutoencoder(3, 6, torch.tensor([1.0, 2.0, 3.0]))
    assert snapshot == str(model), "Model string representation has changed."
