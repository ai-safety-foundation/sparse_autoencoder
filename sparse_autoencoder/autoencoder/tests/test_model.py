"""Sparse Autoencoder Model Tests."""
import os
from pathlib import Path
import uuid

from jaxtyping import Float
import pytest
from syrupy.session import SnapshotSession
import torch
from torch import Tensor
from torch.nn import Parameter
import wandb

from sparse_autoencoder.autoencoder.model import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    SparseAutoencoderState,
)
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


def test_initialize_tied_bias() -> None:
    """Check the tied bias is initialised correctly."""
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=None)
    model = SparseAutoencoder(config, geometric_median)
    assert torch.allclose(model.tied_bias, geometric_median)


def test_encoded_decoded_shape_same() -> None:
    """Check the input and output are the same shape."""
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=None)
    model = SparseAutoencoder(config)
    input_tensor = torch.tensor([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    output = model(input_tensor)

    assert output[1].shape == input_tensor.shape


def test_can_get_encoder_weights() -> None:
    """Check we can access the encoder weights."""
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=None)
    model = SparseAutoencoder(config)
    encoder = model.encoder
    assert encoder.weight.shape == (6, 3)


def test_representation(snapshot: SnapshotSession) -> None:
    """Check the string representation of the model."""
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=None)
    model = SparseAutoencoder(config)
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
    config = SparseAutoencoderConfig(
        n_input_features=n_input_features, n_learned_features=n_learned_features, n_components=None
    )
    model = SparseAutoencoder(config)

    input_tensor = torch.randn(
        shape_with_optional_dimensions(batch_size, n_components, n_input_features)
    )

    model.forward(input_tensor)


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
    config = SparseAutoencoderConfig(
        n_input_features=n_input_features, n_learned_features=n_learned_features, n_components=None
    )
    model_without_components = SparseAutoencoder(config)

    config_with_1_component = SparseAutoencoderConfig(
        n_input_features=n_input_features, n_learned_features=n_learned_features, n_components=1
    )
    model_with_1_component = SparseAutoencoder(config_with_1_component)

    model_with_1_component.encoder.weight = Parameter(
        model_without_components.encoder.weight.unsqueeze(0)
    )
    model_with_1_component.decoder.weight = Parameter(
        model_without_components.decoder.weight.unsqueeze(0)
    )
    model_with_1_component.encoder.bias = Parameter(
        model_without_components.encoder.bias.unsqueeze(0)
    )

    # Forward pass
    output_without_components = model_without_components.forward(input_activations)
    output_with_1_component = model_with_1_component.forward(input_single_component)

    assert torch.allclose(output_with_1_component[0].squeeze(1), output_without_components[0])


def test_save() -> None:
    """Check that the save method stores the config and state dict."""
    # Create the model
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=2)
    model = SparseAutoencoder(config)

    # Save
    path = Path(__file__).parents[3] / ".cache" / "test_save.pt"
    model.save(path)

    # Load the saved file
    serialized_state = torch.load(path)
    state = SparseAutoencoderState.model_validate(serialized_state)

    for key in state.state_dict:
        assert torch.allclose(state.state_dict[key], model.state_dict()[key])

    path.unlink()


def test_load_all_components() -> None:
    """Test loading all components."""
    # Create the model
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=2)
    model = SparseAutoencoder(config)

    # Save
    path = Path(__file__).parents[3] / ".cache" / f"{uuid.uuid4()!s}.pt"
    model.save(path)

    # Load into a new model
    loaded_model = SparseAutoencoder.load(path)

    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], loaded_model.state_dict()[key])

    path.unlink()


def test_load_single_component() -> None:
    """Test loading a single component."""
    # Create the model
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=2)
    model = SparseAutoencoder(config)

    # Save
    path = Path(__file__).parents[3] / ".cache" / f"{uuid.uuid4()!s}.pt"
    model.save(path)

    # Load into a new model
    component_idx = 0
    loaded_model = SparseAutoencoder.load(path, component_idx)

    for key in model.state_dict():
        assert torch.allclose(
            model.state_dict()[key][component_idx], loaded_model.state_dict()[key]
        )

    path.unlink()


@pytest.mark.skipif(os.getenv("WANDB_API_KEY") is None, reason="No wandb API key.")
@pytest.mark.integration_test()
def test_save_load_wandb() -> None:
    """Test saving and loading from wandb."""
    wandb.init(project="test", settings=wandb.Settings(silent=True))

    # Create the model
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=2)
    model = SparseAutoencoder(config)

    # Save
    saved_name = model.save_to_wandb("test")

    # Get it back
    loaded_model = SparseAutoencoder.load_from_wandb(saved_name)

    # Check the state dict is the same
    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], loaded_model.state_dict()[key])

    wandb.finish()


@pytest.mark.skipif(os.getenv("HF_TESTING_ACCESS_TOKEN") is None, reason="No HF access token.")
@pytest.mark.integration_test()
def test_save_load_hugging_face() -> None:
    """Test saving and loading from Hugging Fae."""
    # Create the model
    config = SparseAutoencoderConfig(n_input_features=3, n_learned_features=6, n_components=2)
    model = SparseAutoencoder(config)

    # Save
    file_name = "test-model.pt"
    repo_id = "alancooney/test"
    access_token = os.getenv("HF_TESTING_ACCESS_TOKEN")
    model.save_to_hugging_face(file_name=file_name, repo_id=repo_id, hf_access_token=access_token)

    # Get it back
    loaded_model = SparseAutoencoder.load_from_hugging_face(file_name=file_name, repo_id=repo_id)

    # Check the state dict is the same
    for key in model.state_dict():
        assert torch.allclose(model.state_dict()[key], loaded_model.state_dict()[key])
