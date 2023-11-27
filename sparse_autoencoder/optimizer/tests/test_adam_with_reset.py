"""Tests for AdamWithReset optimizer."""
import pytest
import torch

from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset


@pytest.fixture()
def model_and_optimizer() -> tuple[torch.nn.Module, AdamWithReset]:
    """Model and optimizer fixture."""
    torch.random.manual_seed(0)
    model = SparseAutoencoder(5, 10, torch.rand(5))
    optimizer = AdamWithReset(
        model.parameters(),
        named_parameters=model.named_parameters(),
    )

    # Initialise adam state by doing some steps
    for _ in range(3):
        optimizer.zero_grad()
        _, decoded_activations = model(torch.rand((100, 5)) * 100)
        dummy_loss = (
            torch.nn.functional.mse_loss(
                decoded_activations, torch.rand((100, 5)), reduce=True, reduction="mean"
            )
            * 0.1
        )
        dummy_loss.backward()
        optimizer.step()

    return model, optimizer


def test_initialization(model_and_optimizer: tuple[torch.nn.Module, AdamWithReset]) -> None:
    """Test initialization of AdamWithReset optimizer."""
    model, optimizer = model_and_optimizer
    assert len(optimizer.parameter_names) == len(list(model.named_parameters()))


def test_reset_state_all_parameters(
    model_and_optimizer: tuple[torch.nn.Module, AdamWithReset]
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


def test_reset_neurons_state(model_and_optimizer: tuple[torch.nn.Module, AdamWithReset]) -> None:
    """Test reset_neurons_state method."""
    model, optimizer = model_and_optimizer

    res = optimizer.state[model.encoder.weight]

    # Example usage
    optimizer.reset_neurons_state("_encoder._weight", torch.tensor([1]), axis=0)

    res = optimizer.state[model.encoder.weight]

    assert torch.all(res["exp_avg"][1, :] == 0)
    assert not torch.all(res["exp_avg"][2, :] == 0)
    assert not torch.all(res["exp_avg"][:, 1] == 0)
