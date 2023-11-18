"""Linear encoder tests."""
import pytest
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.autoencoder.components.linear_encoder import LinearEncoder


# Constants for testing
INPUT_FEATURES = 10
LEARNT_FEATURES = 5
BATCH_SIZE = 2


@pytest.fixture()
def encoder() -> LinearEncoder:
    """Fixture to create a LinearEncoder instance."""
    return LinearEncoder(input_features=INPUT_FEATURES, learnt_features=LEARNT_FEATURES)


def test_reset_parameters(encoder: LinearEncoder) -> None:
    """Test resetting of parameters."""
    old_weight = encoder.weight.clone()
    old_bias = encoder.bias.clone()
    encoder.reset_parameters()
    assert not torch.equal(encoder.weight, old_weight)
    assert not torch.equal(encoder.bias, old_bias)


def test_forward_pass(encoder: LinearEncoder) -> None:
    """Test the forward pass of the LinearEncoder."""
    input_tensor = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    output = encoder.forward(input_tensor)
    assert output.shape == (BATCH_SIZE, LEARNT_FEATURES)


def test_extra_repr(encoder: LinearEncoder, snapshot: SnapshotSession) -> None:
    """Test the string representation of the LinearEncoder."""
    assert snapshot == str(encoder), "Model string representation has changed."
