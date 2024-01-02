"""Test the abstract decoder."""

from typing import final

from jaxtyping import Float, Int64
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter, init

from sparse_autoencoder.autoencoder.abstract_autoencoder import ResetOptimizerParameterDetails
from sparse_autoencoder.autoencoder.components.abstract_decoder import AbstractDecoder
from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis


DEFAULT_N_LEARNT_FEATURES = 3
DEFAULT_N_DECODED_FEATURES = 4
DEFAULT_N_COMPONENTS = 2


@final
class MockDecoder(AbstractDecoder):
    """Mock implementation of AbstractDecoder for testing purposes."""

    _weight: Float[Parameter, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]

    def __init__(
        self,
        learnt_features: int = DEFAULT_N_LEARNT_FEATURES,
        decoded_features: int = DEFAULT_N_DECODED_FEATURES,
        n_components: int = DEFAULT_N_COMPONENTS,
    ) -> None:
        """Initialise the mock decoder."""
        super().__init__(
            decoded_features=decoded_features, learnt_features=learnt_features, n_components=None
        )
        torch.random.manual_seed(0)
        self._weight = Parameter(torch.empty(n_components, decoded_features, learnt_features))

    @property
    def weight(
        self,
    ) -> Float[
        Parameter, Axis.names(Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE)
    ]:
        """Get the weight of the decoder."""
        return self._weight

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details."""
        return [ResetOptimizerParameterDetails(parameter=self.weight, axis=1)]

    def forward(
        self, x: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)]:
        """Mock forward pass."""
        return torch.nn.functional.linear(x, self.weight)

    def reset_parameters(self) -> None:
        """Mock reset parameters."""
        init.kaiming_normal_(self._weight)

    def constrain_weights_unit_norm(self) -> None:
        """Constrain weights."""


@pytest.fixture()
def mock_decoder() -> MockDecoder:
    """Pytest fixture to provide a MockDecoder instance."""
    return MockDecoder()


def test_update_dictionary_vectors_with_no_neurons(mock_decoder: MockDecoder) -> None:
    """Test update_dictionary_vectors with 0 neurons to update."""
    original_weight = mock_decoder.weight.clone()  # Save original weight for comparison

    dictionary_vector_indices: Int64[
        Tensor, Axis.names(Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
    ] = torch.empty((0, 0), dtype=torch.int64)

    updates: Float[
        Tensor, Axis.names(Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)
    ] = torch.empty((0, 0, 0), dtype=torch.float)

    mock_decoder.update_dictionary_vectors(dictionary_vector_indices, updates)

    # Ensure weight did not change when no indices were provided
    assert torch.equal(
        mock_decoder.weight, original_weight
    ), "Weights should not change when no indices are provided."


@pytest.mark.parametrize(
    ("dictionary_vector_indices", "updates"),
    [
        pytest.param(torch.tensor([1]), torch.rand(4, 1), id="One neuron to update"),
        pytest.param(
            torch.tensor([0, 2]),
            torch.rand(4, 2),
            id="Two neurons to update",
        ),
    ],
)
def test_update_dictionary_vectors_with_neurons(
    mock_decoder: MockDecoder,
    dictionary_vector_indices: Int64[Tensor, Axis.INPUT_OUTPUT_FEATURE],
    updates: Float[Tensor, Axis.names(Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)],
) -> None:
    """Test update_dictionary_vectors with 1 or 2 neurons to update."""
    mock_decoder.update_dictionary_vectors(dictionary_vector_indices, updates, component_idx=0)

    # Check if the specified neurons are updated correctly
    assert torch.allclose(
        mock_decoder.weight[0, :, dictionary_vector_indices], updates
    ), "update_dictionary_vectors should update the weights correctly."
