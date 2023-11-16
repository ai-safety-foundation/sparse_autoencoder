"""Tests for LearnedActivationsL1Loss."""
import pytest
import torch

from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss


@pytest.fixture()
def l1_loss() -> LearnedActivationsL1Loss:
    """Fixture for LearnedActivationsL1Loss with a default L1 coefficient."""
    return LearnedActivationsL1Loss(l1_coefficient=0.1)


def test_l1_loss_forward(l1_loss: LearnedActivationsL1Loss) -> None:
    """Test the forward method of LearnedActivationsL1Loss."""
    learned_activations = torch.tensor([[2.0, -3.0], [2.0, -3.0]])
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    expected_loss = torch.tensor([0.5, 0.5])  # (|2| + |-3|) * 0.1 for each row
    calculated_loss = l1_loss.forward(source_activations, learned_activations, decoded_activations)

    assert torch.allclose(calculated_loss, expected_loss), "L1 loss calculation is incorrect."


def test_l1_loss_with_different_l1_coefficients() -> None:
    """Test LearnedActivationsL1Loss with different L1 coefficients."""
    learned_activations = torch.tensor([[2.0, -3.0], [2.0, -3.0]])
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    for coefficient in [0.01, 0.1, 0.5]:
        l1_loss = LearnedActivationsL1Loss(l1_coefficient=coefficient)
        expected_loss = torch.abs(learned_activations).sum(dim=-1) * coefficient
        calculated_loss = l1_loss.forward(
            source_activations, learned_activations, decoded_activations
        )

        assert torch.allclose(
            calculated_loss, expected_loss
        ), f"L1 loss calculation is incorrect for coefficient {coefficient}."


def test_l1_loss_with_zero_input(l1_loss: LearnedActivationsL1Loss) -> None:
    """Test the L1 loss function with zero inputs."""
    learned_activations = torch.zeros((2, 3))
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    expected_loss = torch.zeros(2)
    calculated_loss = l1_loss.forward(source_activations, learned_activations, decoded_activations)

    assert torch.all(calculated_loss == expected_loss), "L1 loss should be zero for zero inputs."


def test_l1_loss_with_negative_input(l1_loss: LearnedActivationsL1Loss) -> None:
    """Test the L1 loss function with negative inputs."""
    learned_activations = torch.tensor([[-2.0, -3.0], [-1.0, -4.0]])
    source_activations = decoded_activations = torch.zeros_like(learned_activations)

    expected_loss = torch.tensor([0.5, 0.5])  # (|2| + |-3|) * 0.1 for each row
    calculated_loss = l1_loss.forward(source_activations, learned_activations, decoded_activations)

    assert torch.allclose(
        calculated_loss, expected_loss
    ), "L1 loss calculation is incorrect with negative inputs."
