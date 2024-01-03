"""Linear encoder tests."""
from jaxtyping import Float, Int64
import pytest
from syrupy.session import SnapshotSession
import torch
from torch import Tensor

from sparse_autoencoder.autoencoder.components.linear_encoder import LinearEncoder
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


# Constants for testing
INPUT_FEATURES = 2
LEARNT_FEATURES = 4
N_COMPONENTS = 2
BATCH_SIZE = 3


@pytest.fixture()
def encoder() -> LinearEncoder:
    """Fixture to create a LinearEncoder instance."""
    torch.manual_seed(0)
    return LinearEncoder(
        input_features=INPUT_FEATURES, learnt_features=LEARNT_FEATURES, n_components=N_COMPONENTS
    )


def test_reset_parameters(encoder: LinearEncoder) -> None:
    """Test resetting of parameters."""
    old_weight = encoder.weight.clone()
    old_bias = encoder.bias.clone()
    encoder.reset_parameters()
    assert not torch.equal(encoder.weight, old_weight)
    assert not torch.equal(encoder.bias, old_bias)


def test_forward_pass(encoder: LinearEncoder) -> None:
    """Test the forward pass of the LinearEncoder."""
    input_tensor = torch.randn(BATCH_SIZE, N_COMPONENTS, INPUT_FEATURES)
    output = encoder.forward(input_tensor)
    assert output.shape == (BATCH_SIZE, N_COMPONENTS, LEARNT_FEATURES)


def test_extra_repr(encoder: LinearEncoder, snapshot: SnapshotSession) -> None:
    """Test the string representation of the LinearEncoder."""
    assert snapshot == str(encoder), "Model string representation has changed."


@pytest.mark.parametrize("n_components", [None, 1, 3])
def test_forward_pass_result_matches_the_snapshot(
    n_components: int | None, snapshot: SnapshotSession
) -> None:
    """Test the forward pass of the LinearEncoder."""
    torch.manual_seed(1)
    input_tensor = torch.rand(
        shape_with_optional_dimensions(BATCH_SIZE, n_components, INPUT_FEATURES)
    )
    encoder = LinearEncoder(
        input_features=INPUT_FEATURES, learnt_features=LEARNT_FEATURES, n_components=n_components
    )
    output = encoder.forward(input_tensor)
    assert snapshot == output


def test_output_same_without_component_dim_vs_with_1_component() -> None:
    """Test the forward pass gives identical results for None and 1 component."""
    # Create the layers to compare
    encoder_without_components_dim = LinearEncoder(
        input_features=INPUT_FEATURES, learnt_features=LEARNT_FEATURES, n_components=None
    )
    encoder_with_1_component = LinearEncoder(
        input_features=INPUT_FEATURES, learnt_features=LEARNT_FEATURES, n_components=1
    )

    # Set the weight and value parameters to be the same
    encoder_with_1_component._weight = torch.nn.Parameter(  # type: ignore # noqa: SLF001
        encoder_without_components_dim._weight.unsqueeze(0)  # type: ignore # noqa: SLF001
    )
    encoder_with_1_component._bias = torch.nn.Parameter(  # type: ignore # noqa: SLF001
        encoder_without_components_dim._bias.unsqueeze(0)  # type: ignore # noqa: SLF001
    )

    # Create the input
    input_tensor = torch.rand(BATCH_SIZE, INPUT_FEATURES)
    input_with_components_dim = input_tensor.unsqueeze(1)

    # Check the output is the same
    output_without_components_dim = encoder_without_components_dim(input_tensor)
    output_with_1_component = encoder_with_1_component(input_with_components_dim)

    assert torch.allclose(output_without_components_dim, output_with_1_component.squeeze(1))


def test_update_dictionary_vectors_with_no_neurons(encoder: LinearEncoder) -> None:
    """Test update_dictionary_vectors with 0 neurons to update."""
    torch.random.manual_seed(0)
    original_weight = encoder.weight.clone()  # Save original weight for comparison

    dictionary_vector_indices: Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)] = torch.empty(
        0, dtype=torch.int64
    )

    updates: Float[
        Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX, Axis.INPUT_OUTPUT_FEATURE)
    ] = torch.empty((0, INPUT_FEATURES), dtype=torch.float)

    encoder.update_dictionary_vectors(dictionary_vector_indices, updates, component_idx=0)

    # Ensure weight did not change when no indices were provided
    assert torch.equal(
        encoder.weight, original_weight
    ), "Weights should not change when no indices are provided."


@pytest.mark.parametrize(
    ("dictionary_vector_indices", "updates"),
    [
        pytest.param(torch.tensor([1]), torch.rand((1, INPUT_FEATURES)), id="update 1 neuron"),
        pytest.param(
            torch.tensor([0, 1]),
            torch.rand((2, INPUT_FEATURES)),
            id="update 2 neurons with different values",
        ),
    ],
)
def test_update_dictionary_vectors_with_neurons(
    encoder: LinearEncoder,
    dictionary_vector_indices: Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)],
    updates: Float[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX, Axis.INPUT_OUTPUT_FEATURE)],
) -> None:
    """Test update_dictionary_vectors with 1 or 2 neurons to update."""
    with torch.no_grad():
        component_idx = 0
        encoder.update_dictionary_vectors(
            dictionary_vector_indices, updates, component_idx=component_idx
        )

        # Check if the specified neurons are updated correctly
        assert torch.allclose(
            encoder.weight[component_idx, dictionary_vector_indices, :], updates
        ), "update_dictionary_vectors should update the weights correctly."
