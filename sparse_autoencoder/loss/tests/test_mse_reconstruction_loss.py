"""Tests for the MSEReconstructionLoss class."""
import pytest
import torch

from sparse_autoencoder.loss.mse_reconstruction_loss import MSEReconstructionLoss


@pytest.fixture()
def mse_loss() -> MSEReconstructionLoss:
    """Fixture for MSEReconstructionLoss."""
    return MSEReconstructionLoss()


def test_mse_loss_forward(mse_loss: MSEReconstructionLoss) -> None:
    """Test the forward method of MSEReconstructionLoss."""
    input_activations = torch.tensor([[5.0, 4.0], [3.0, 4.0]])
    output_activations = torch.tensor([[1.0, 5.0], [1.0, 5.0]])
    learned_activations = torch.zeros_like(input_activations)

    expected_loss = torch.tensor([8.5, 2.5])
    calculated_loss = mse_loss.forward(input_activations, learned_activations, output_activations)

    assert torch.allclose(calculated_loss, expected_loss), "MSE loss calculation is incorrect."


def test_mse_loss_with_zero_input(mse_loss: MSEReconstructionLoss) -> None:
    """Test the MSE loss function with zero inputs."""
    input_activations = torch.zeros((2, 3))
    output_activations = torch.zeros_like(input_activations)
    learned_activations = torch.zeros_like(input_activations)

    expected_loss = torch.zeros(2)
    calculated_loss = mse_loss.forward(input_activations, learned_activations, output_activations)

    assert torch.all(
        calculated_loss == expected_loss
    ), "MSE loss should be zero for identical zero inputs."


def test_mse_loss_with_varying_input_shapes(mse_loss: MSEReconstructionLoss) -> None:
    """Test the MSE loss function with varying input shapes."""
    for shape in [(1, 3), (5, 3), (10, 5)]:
        input_activations = torch.rand(shape)
        output_activations = torch.rand(shape)
        learned_activations = torch.zeros_like(input_activations)

        calculated_loss = mse_loss.forward(
            input_activations, learned_activations, output_activations
        )

        # Just checking if the loss calculation completes without error for different shapes
        assert (
            calculated_loss.shape[0] == shape[0]
        ), f"MSE loss calculation failed for shape {shape}."
