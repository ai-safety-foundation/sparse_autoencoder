"""Sparse Autoencoder Model Tests."""
from jaxtyping import Float
import pytest
from syrupy.session import SnapshotSession
import torch
from torch import Tensor

from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.autoencoder.utils.tensor_shape import shape_with_optional_dimensions
from sparse_autoencoder.tensor_types import Axis


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
def test_forward_pass_result_matches_the_snapshot(
    n_components: int | None, snapshot: SnapshotSession
) -> None:
    """Check the forward pass works without errors."""
    batch_size = 3
    n_input_features = 6
    n_learned_features = 12

    torch.manual_seed(1)
    model = SparseAutoencoder(n_input_features, n_learned_features, n_components=n_components)

    input_tensor = torch.randn(
        shape_with_optional_dimensions(batch_size, n_components, n_input_features)
    )

    output = model.forward(input_tensor)
    assert output[1] == snapshot, "Forward pass result has changed."


def test_forward_pass_same_without_components_and_1_component() -> None:
    """Test the forward pass gives identical results for None and 1 component."""
    batch_size = 3
    n_input_features = 6
    n_learned_features = 12

    # Inputs
    input_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)
    ] = torch.randn(batch_size, n_input_features)
    input_single_component: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
    ] = input_activations.unsqueeze(1)

    # Create the models
    torch.manual_seed(1)
    model_without_components = SparseAutoencoder(
        n_input_features, n_learned_features, n_components=None
    )
    torch.manual_seed(1)
    model_with_1_component = SparseAutoencoder(n_input_features, n_learned_features, n_components=1)

    # Forward pass
    output_without_components = model_without_components.forward(input_activations)
    output_with_1_component = model_with_1_component.forward(input_single_component)

    assert torch.allclose(output_with_1_component[0].squeeze(1), output_without_components[0])
