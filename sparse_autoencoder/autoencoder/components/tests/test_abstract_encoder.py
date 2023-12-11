"""Test the abstract encoder."""

from typing import final

from jaxtyping import Float, Int64
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter, init

from sparse_autoencoder.autoencoder.components.abstract_encoder import (
    AbstractEncoder,
)
from sparse_autoencoder.tensor_types import Axis


@final
class MockEncoder(AbstractEncoder):
    """Mock implementation of AbstractEncoder for testing purposes."""

    def __init__(self, encoded_features: int = 4, learnt_features: int = 3) -> None:
        """Initialise."""
        super().__init__()
        torch.random.manual_seed(0)
        self._weight: Parameter = Parameter(torch.empty(learnt_features, encoded_features))
        self._bias: Parameter = Parameter(torch.empty(encoded_features))

    @property
    def weight(
        self,
    ) -> Float[Parameter, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]:
        """Get the weight of the encoder."""
        return self._weight

    @property
    def bias(self) -> Float[Parameter, Axis.LEARNT_FEATURE]:
        """Get the bias of the encoder."""
        return self._bias

    def forward(
        self, x: Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)]
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]:
        """Mock forward pass."""
        return torch.nn.functional.linear(x, self.weight)

    def reset_parameters(self) -> None:
        """Mock reset parameters."""
        self._weight: Parameter = init.kaiming_normal_(self._weight)  # type: ignore

    @property
    def reset_optimizer_parameter_details(self) -> list[tuple[Parameter, int]]:
        """Reset optimizer parameter details."""
        return [(self.weight, 0), (self.bias, 0)]


@pytest.fixture()
def mock_encoder() -> MockEncoder:
    """Pytest fixture to provide a MockEncoder instance."""
    return MockEncoder()


def test_forward_method(mock_encoder: MockEncoder) -> None:
    """Test the forward method of the encoder runs without errors."""
    assert callable(mock_encoder.forward), "Forward method should be callable."


def test_reset_parameters_method(mock_encoder: MockEncoder) -> None:
    """Test the reset_parameters method of the encoder runs without errors."""
    assert callable(mock_encoder.reset_parameters), "Reset_parameters method should be callable."


def test_update_dictionary_vectors_with_no_neurons(mock_encoder: MockEncoder) -> None:
    """Test update_dictionary_vectors with 0 neurons to update."""
    torch.random.manual_seed(0)
    original_weight = mock_encoder.weight.clone()  # Save original weight for comparison

    dictionary_vector_indices: Int64[Tensor, Axis.INPUT_OUTPUT_FEATURE] = torch.empty(
        0,
        dtype=torch.int64,  # Empty tensor with 1 dimension
    )
    updates: Float[Tensor, Axis.INPUT_OUTPUT_FEATURE] = torch.empty(
        (0, 0),
        dtype=torch.float,  # Empty tensor with 2 dimensions
    )

    mock_encoder.update_dictionary_vectors(dictionary_vector_indices, updates)

    # Ensure weight did not change when no indices were provided
    assert torch.equal(
        mock_encoder.weight, original_weight
    ), "Weights should not change when no indices are provided."


@pytest.mark.parametrize(
    ("dictionary_vector_indices", "updates"),
    [
        (torch.tensor([1]), torch.rand((1, 4))),  # Test with 1 neuron to update
        (
            torch.tensor([0, 2]),
            torch.rand((2, 4)),
        ),  # Test with 2 neurons to update
    ],
)
def test_update_dictionary_vectors_with_neurons(
    mock_encoder: MockEncoder,
    dictionary_vector_indices: Int64[Tensor, Axis.INPUT_OUTPUT_FEATURE],
    updates: Float[Tensor, Axis.INPUT_OUTPUT_FEATURE],
) -> None:
    """Test update_dictionary_vectors with 1 or 2 neurons to update."""
    mock_encoder.update_dictionary_vectors(dictionary_vector_indices, updates)

    # Check if the specified neurons are updated correctly
    assert torch.allclose(
        mock_encoder.weight[dictionary_vector_indices, :], updates
    ), "update_dictionary_vectors should update the weights correctly."
