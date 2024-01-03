"""Tests for the constrained unit norm linear layer."""
from jaxtyping import Float, Int64
import pytest
import torch
from torch import Tensor

from sparse_autoencoder.autoencoder.components.unit_norm_decoder import UnitNormDecoder
from sparse_autoencoder.tensor_types import Axis


DEFAULT_N_LEARNT_FEATURES = 3
DEFAULT_N_DECODED_FEATURES = 4
DEFAULT_N_COMPONENTS = 2


@pytest.fixture()
def decoder() -> UnitNormDecoder:
    """Pytest fixture to provide a MockDecoder instance."""
    return UnitNormDecoder(
        learnt_features=DEFAULT_N_LEARNT_FEATURES,
        decoded_features=DEFAULT_N_DECODED_FEATURES,
        n_components=DEFAULT_N_COMPONENTS,
    )


def test_initialization() -> None:
    """Test that the weights are initialized with unit norm."""
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4, n_components=None)
    weight_norms = torch.norm(layer.weight, dim=0)
    assert torch.allclose(weight_norms, torch.ones_like(weight_norms))


def test_forward_pass() -> None:
    """Test the forward pass of the layer."""
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4, n_components=None)
    input_tensor = torch.randn(5, 3)  # Batch size of 5, learnt features of 3
    output = layer(input_tensor)
    assert output.shape == (5, 4)  # Batch size of 5, decoded features of 4


def test_multiple_training_steps() -> None:
    """Test the unit norm constraint over multiple training iterations."""
    torch.random.manual_seed(0)
    layer = UnitNormDecoder(learnt_features=3, decoded_features=4, n_components=None)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(4):
        data = torch.randn(5, 3)
        optimizer.zero_grad()
        logits = layer(data)

        loss = torch.mean(logits**2)
        loss.backward()
        optimizer.step()
        layer.constrain_weights_unit_norm()

        columns_norms = torch.norm(layer.weight, dim=0)
        assert torch.allclose(columns_norms, torch.ones_like(columns_norms))


def test_unit_norm_decreases() -> None:
    """Check that the unit norm is applied after each gradient step."""
    for _ in range(4):
        data = torch.randn((1, 3), requires_grad=True)

        # run with the backward hook
        layer = UnitNormDecoder(learnt_features=3, decoded_features=4, n_components=None)
        layer_weights = torch.nn.Parameter(layer.weight.clone())
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0)
        logits = layer(data)
        loss = torch.mean(logits**2)
        loss.backward()
        optimizer.step()
        weight_norms_with_hook = torch.sum(layer.weight**2, dim=0).clone()

        # Run without the hook
        layer_without_hook = UnitNormDecoder(
            learnt_features=3, decoded_features=4, n_components=None, enable_gradient_hook=False
        )
        layer_without_hook._weight = layer_weights  # type: ignore (testing only)  # noqa: SLF001
        optimizer_without_hook = torch.optim.SGD(
            layer_without_hook.parameters(), lr=0.1, momentum=0
        )
        logits_without_hook = layer_without_hook(data)
        loss_without_hook = torch.mean(logits_without_hook**2)
        loss_without_hook.backward()
        optimizer_without_hook.step()
        weight_norms_without_hook = torch.sum(layer_without_hook.weight**2, dim=0).clone()

        # Check that the norm with the hook is closer to 1 than without the hook
        target_norms = torch.ones_like(weight_norms_with_hook)
        absolute_diff_with_hook = torch.abs(weight_norms_with_hook - target_norms)
        absolute_diff_without_hook = torch.abs(weight_norms_without_hook - target_norms)
        assert torch.all(absolute_diff_with_hook < absolute_diff_without_hook)


def test_output_same_without_component_dim_vs_with_1_component() -> None:
    """Test the forward pass gives identical results for None and 1 component."""
    decoded_features = 2
    learnt_features = 4
    batch_size = 1

    # Create the layers to compare
    torch.manual_seed(1)
    decoder_without_components_dim = UnitNormDecoder(
        decoded_features=decoded_features, learnt_features=learnt_features, n_components=None
    )
    torch.manual_seed(1)
    decoder_with_1_component = UnitNormDecoder(
        decoded_features=decoded_features, learnt_features=learnt_features, n_components=1
    )

    # Create the input
    input_tensor = torch.randn(batch_size, learnt_features)
    input_with_components_dim = input_tensor.unsqueeze(1)

    # Check the output is the same
    output_without_components_dim = decoder_without_components_dim(input_tensor)
    output_with_1_component = decoder_with_1_component(input_with_components_dim)
    assert torch.allclose(output_without_components_dim, output_with_1_component.squeeze(1))


def test_update_dictionary_vectors_with_no_neurons(decoder: UnitNormDecoder) -> None:
    """Test update_dictionary_vectors with 0 neurons to update."""
    original_weight = decoder.weight.clone()  # Save original weight for comparison

    dictionary_vector_indices: Int64[
        Tensor, Axis.names(Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
    ] = torch.empty((0, 0), dtype=torch.int64)

    updates: Float[
        Tensor, Axis.names(Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)
    ] = torch.empty((0, 0, 0), dtype=torch.float)

    decoder.update_dictionary_vectors(dictionary_vector_indices, updates)

    # Ensure weight did not change when no indices were provided
    assert torch.equal(
        decoder.weight, original_weight
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
    decoder: UnitNormDecoder,
    dictionary_vector_indices: Int64[Tensor, Axis.INPUT_OUTPUT_FEATURE],
    updates: Float[Tensor, Axis.names(Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)],
) -> None:
    """Test update_dictionary_vectors with 1 or 2 neurons to update."""
    decoder.update_dictionary_vectors(dictionary_vector_indices, updates, component_idx=0)

    # Check if the specified neurons are updated correctly
    assert torch.allclose(
        decoder.weight[0, :, dictionary_vector_indices], updates
    ), "update_dictionary_vectors should update the weights correctly."
