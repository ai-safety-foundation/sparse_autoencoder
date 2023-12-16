"""Tests for AdamWithReset optimizer."""
import pytest
import torch

from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset


@pytest.fixture()
def model_and_optimizer() -> tuple[torch.nn.Module, AdamWithReset]:
    """Model and optimizer fixture."""
    torch.random.manual_seed(0)
    model = SparseAutoencoder(5, 10)
    optimizer = AdamWithReset(
        model.parameters(), named_parameters=model.named_parameters(), lr=0.0001
    )

    # Initialise adam state by doing some steps
    for _ in range(3):
        source_activations = torch.rand((10, 5))
        optimizer.zero_grad()
        _, decoded_activations = model.forward(source_activations)
        dummy_loss = torch.nn.functional.mse_loss(decoded_activations, source_activations)
        dummy_loss.backward()
        optimizer.step()

    # Force all state values to be non-zero
    optimizer.state[model.encoder.weight]["exp_avg"] = (
        optimizer.state[model.encoder.weight]["exp_avg"] + 1.0
    )
    optimizer.state[model.encoder.weight]["exp_avg_sq"] = (
        optimizer.state[model.encoder.weight]["exp_avg_sq"] + 1.0
    )

    return model, optimizer


def test_initialization(model_and_optimizer: tuple[torch.nn.Module, AdamWithReset]) -> None:
    """Test initialization of AdamWithReset optimizer."""
    model, optimizer = model_and_optimizer
    assert len(optimizer.parameter_names) == len(list(model.named_parameters()))


def test_reset_state_all_parameters(
    model_and_optimizer: tuple[torch.nn.Module, AdamWithReset],
) -> None:
    """Test reset_state_all_parameters method."""
    _, optimizer = model_and_optimizer
    optimizer.reset_state_all_parameters()

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            # Get the state
            parameter_state = optimizer.state[parameter]
            for state_name in parameter_state:
                if state_name in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
                    # Check all state values are reset to zero
                    state = parameter_state[state_name]
                    assert torch.all(state == 0)


def test_zero_fill_multi_dimensional_same_index_fill_2d(
    model_and_optimizer: tuple[torch.nn.Module, AdamWithReset],
) -> None:
    """Test zero_fill_multi_dimensional is the same as using index fill for 2d tensors."""
    _model, optimizer = model_and_optimizer
    weights_tensor = torch.rand((10, 5))
    indices = torch.tensor([1, 2, 3])
    axis = 0

    comparison = weights_tensor.index_fill(axis, indices, 0)
    res = optimizer.zero_fill_multi_dimensional(weights_tensor, indices, axis)

    assert torch.all(res == comparison)


@pytest.mark.parametrize(
    ("weights_dimensions", "indices", "axis"),
    [
        ((10, 5), torch.tensor([1, 2, 3]), 0),
    ],
)
def test_zero_fill_multi_dimensional(
    model_and_optimizer: tuple[torch.nn.Module, AdamWithReset],
    weights_dimensions: tuple[int],
    indices: torch.Tensor,
    axis: int,
) -> None:
    """Test zero_fill_multi_dimensional."""
    _model, optimizer = model_and_optimizer

    res = optimizer.zero_fill_multi_dimensional(torch.rand(weights_dimensions), indices, axis)

    for row in range(weights_dimensions[axis]):
        for col in range(weights_dimensions[1 - axis]):
            if row in indices:
                assert res[row, col] == 0
            else:
                assert res[row, col] != 0


def test_reset_neurons_state(model_and_optimizer: tuple[torch.nn.Module, AdamWithReset]) -> None:
    """Test reset_neurons_state method."""
    model, optimizer = model_and_optimizer
    optimizer.reset_neurons_state(model.encoder.weight, torch.tensor([1]), axis=0)

    res = optimizer.state[model.encoder.weight]

    assert torch.all(res["exp_avg"][1, :] == 0)
    assert not torch.any(res["exp_avg"][2, :] == 0)
    assert not torch.any(res["exp_avg"][2:, 1] == 0)


def test_reset_neurons_state_no_dead_neurons(
    model_and_optimizer: tuple[torch.nn.Module, AdamWithReset],
) -> None:
    """Test reset_neurons_state method with 0 dead neurons."""
    model, optimizer = model_and_optimizer

    res = optimizer.state[model.encoder.weight]

    # Example usage
    optimizer.reset_neurons_state(model.encoder.weight, torch.tensor([], dtype=torch.int64), axis=0)

    res = optimizer.state[model.encoder.weight]

    assert not torch.any(res["exp_avg"] == 0)
    assert not torch.any(res["exp_avg_sq"] == 0)
