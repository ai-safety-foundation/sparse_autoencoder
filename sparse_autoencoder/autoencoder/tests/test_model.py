"""Sparse Autoencoder Model Tests."""
import torch

from sparse_autoencoder.autoencoder.model import SparseAutoencoder


def test_initialize_tied_bias():
    """Check the tied bias is initialised correctly."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)
    assert torch.allclose(model.tied_bias, geometric_median)


def test_encoded_decoded_shape_same():
    """Check the input and output are the same shape."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)
    input_tensor = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    output = model(input_tensor)

    assert output[1].shape == input_tensor.shape
