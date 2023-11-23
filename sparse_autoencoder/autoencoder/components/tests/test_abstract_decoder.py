"""Test the abstract decoder."""

from typing import final

import pytest
import torch
from torch.nn import Parameter, init

from sparse_autoencoder.autoencoder.components.abstract_decoder import AbstractDecoder
from sparse_autoencoder.tensor_types import (
    DeadDecoderNeuronWeightUpdates,
    DecoderWeights,
    EncoderWeights,
    InputOutputActivationBatch,
    InputOutputNeuronIndices,
    LearnedActivationBatch,
)


@final
class MockDecoder(AbstractDecoder):
    """Mock implementation of AbstractDecoder for testing purposes."""

    def __init__(self, learnt_features: int = 3, decoded_features: int = 4) -> None:
        """Initialise the mock decoder."""
        super().__init__()
        torch.random.manual_seed(0)
        self._weight = Parameter(torch.empty(decoded_features, learnt_features))

    @property
    def weight(self) -> DecoderWeights:
        """Get the weight of the decoder."""
        return self._weight

    def forward(self, x: LearnedActivationBatch) -> InputOutputActivationBatch:
        """Mock forward pass."""
        return torch.nn.functional.linear(x, self.weight)

    def reset_parameters(self) -> None:
        """Mock reset parameters."""
        self._weight: EncoderWeights = init.kaiming_normal_(
            self._weight,
        )


@pytest.fixture()
def mock_decoder() -> MockDecoder:
    """Pytest fixture to provide a MockDecoder instance."""
    return MockDecoder()


def test_forward_method(mock_decoder: MockDecoder) -> None:
    """Test the forward method of the decoder runs without errors."""
    assert callable(mock_decoder.forward), "Forward method should be callable."


def test_reset_parameters_method(mock_decoder: MockDecoder) -> None:
    """Test the reset_parameters method of the decoder runs without errors."""
    assert callable(mock_decoder.reset_parameters), "Reset_parameters method should be callable."


def test_update_dictionary_vectors_with_no_neurons(mock_decoder: MockDecoder) -> None:
    """Test update_dictionary_vectors with 0 neurons to update."""
    original_weight = mock_decoder.weight.clone()  # Save original weight for comparison

    dictionary_vector_indices: InputOutputNeuronIndices = torch.empty(
        0,
        dtype=torch.int,  # Empty tensor with 1 dimension
    )
    updates: DeadDecoderNeuronWeightUpdates = torch.empty(
        (0, 0),
        dtype=torch.float,  # Empty tensor with 2 dimensions
    )

    mock_decoder.update_dictionary_vectors(dictionary_vector_indices, updates)

    # Ensure weight did not change when no indices were provided
    assert torch.equal(
        mock_decoder.weight, original_weight
    ), "Weights should not change when no indices are provided."


@pytest.mark.parametrize(
    ("dictionary_vector_indices", "updates"),
    [
        (torch.tensor([1]), torch.rand(4, 1)),  # Test with 1 neuron to update
        (
            torch.tensor([0, 2]),
            torch.rand(4, 2),
        ),  # Test with 2 neurons to update
    ],
)
def test_update_dictionary_vectors_with_neurons(
    mock_decoder: MockDecoder,
    dictionary_vector_indices: InputOutputNeuronIndices,
    updates: DeadDecoderNeuronWeightUpdates,
) -> None:
    """Test update_dictionary_vectors with 1 or 2 neurons to update."""
    mock_decoder.update_dictionary_vectors(dictionary_vector_indices, updates)

    # Check if the specified neurons are updated correctly
    assert torch.allclose(
        mock_decoder.weight[:, dictionary_vector_indices], updates
    ), "update_dictionary_vectors should update the weights correctly."
