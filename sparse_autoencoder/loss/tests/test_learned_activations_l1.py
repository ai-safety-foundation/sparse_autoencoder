"""Tests for LearnedActivationsL1Loss."""
from einops import repeat
from jaxtyping import Float
import pytest
import torch
from torch import Tensor, tensor

from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
from sparse_autoencoder.tensor_types import Axis


DEFAULT_N_COMPONENTS: int = 3


@pytest.fixture()
def l1_loss() -> LearnedActivationsL1Loss:
    """Fixture for LearnedActivationsL1Loss with a default L1 coefficient."""
    return LearnedActivationsL1Loss(l1_coefficient=0.1)


def l1_loss_component_l1_coefficient() -> LearnedActivationsL1Loss:
    """Fixture for LearnedActivationsL1Loss with a component-wise L1 coefficient."""
    l1_coefficient: Float[Tensor, Axis.COMPONENT] = torch.arange(1, DEFAULT_N_COMPONENTS + 1) / 10
    return LearnedActivationsL1Loss(l1_coefficient=l1_coefficient)


@pytest.mark.parametrize(
    ("l1_coefficient", "learned_activations", "expected_loss"),
    [
        (
            0.1,
            tensor([[2.0, -3.0], [2.0, -3.0]]),  # (batch, learnt_feature)
            tensor([0.5, 0.5]),
        ),
        # Tests with components dimension
        (
            0.1,
            repeat(
                tensor([[2.0, -3.0], [2.0, -3.0]]),
                "batch feature -> batch component feature",
                component=DEFAULT_N_COMPONENTS,
            ),
            repeat(tensor([0.5, 0.5]), "batch -> batch component", component=DEFAULT_N_COMPONENTS),
        ),
        (
            tensor([0.1, 0.1, 0.1]),
            repeat(
                tensor([[2.0, -3.0], [2.0, -3.0]]),
                "batch feature -> batch component feature",
                component=DEFAULT_N_COMPONENTS,
            ),
            repeat(tensor([0.5, 0.5]), "batch -> batch component", component=DEFAULT_N_COMPONENTS),
        ),
        (
            tensor([0.1, 0.2, 0.3]),
            repeat(
                tensor([[2.0, -3.0], [2.0, -3.0]]),
                "batch feature -> batch component feature",
                component=DEFAULT_N_COMPONENTS,
            ),
            tensor(
                [
                    [0.5, 1.0, 1.5],
                    [0.5, 1.0, 1.5],
                ]
            ),
        ),
    ],
)
def test_l1_loss_forward(
    l1_coefficient: float | Float[Tensor, Axis.COMPONENT_OPTIONAL],
    learned_activations: Tensor,
    expected_loss: Tensor,
) -> None:
    """Test the forward method of LearnedActivationsL1Loss."""
    loss_fn = LearnedActivationsL1Loss(l1_coefficient=l1_coefficient)
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    calculated_loss = loss_fn.forward(source_activations, learned_activations, decoded_activations)

    assert torch.allclose(calculated_loss, expected_loss), "L1 loss calculation is incorrect."


@pytest.mark.parametrize(("l1_coefficient"), [0.01, 0.1, 0.5])
def test_l1_loss_with_different_l1_coefficients(l1_coefficient: float) -> None:
    """Test LearnedActivationsL1Loss with different L1 coefficients."""
    learned_activations = tensor([[2.0, -3.0], [2.0, -3.0]])
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    l1_loss = LearnedActivationsL1Loss(l1_coefficient=l1_coefficient)
    expected_loss = torch.abs(learned_activations).sum(dim=-1) * l1_coefficient
    calculated_loss = l1_loss.forward(source_activations, learned_activations, decoded_activations)

    assert torch.allclose(
        calculated_loss, expected_loss
    ), f"L1 loss calculation is incorrect for coefficient {l1_coefficient}."


def test_l1_loss_with_zero_input(l1_loss: LearnedActivationsL1Loss) -> None:
    """Test the L1 loss function with zero inputs."""
    learned_activations = torch.zeros((2, 3))
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    expected_loss = torch.zeros(2)
    calculated_loss = l1_loss.forward(source_activations, learned_activations, decoded_activations)

    assert torch.all(calculated_loss == expected_loss), "L1 loss should be zero for zero inputs."


def test_l1_loss_with_negative_input(l1_loss: LearnedActivationsL1Loss) -> None:
    """Test the L1 loss function with negative inputs."""
    learned_activations = tensor([[-2.0, -3.0], [-1.0, -4.0]])
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    expected_loss = tensor([0.5, 0.5])  # (|2| + |-3|) * 0.1 for each row
    calculated_loss = l1_loss.forward(source_activations, learned_activations, decoded_activations)

    assert torch.allclose(
        calculated_loss, expected_loss
    ), "L1 loss calculation is incorrect with negative inputs."
