"""Tests for the L2ReconstructionLoss class."""
import pytest
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.loss.decoded_activations_l2 import L2ReconstructionLoss
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


@pytest.fixture()
def l2_loss() -> L2ReconstructionLoss:
    """Fixture for L2ReconstructionLoss."""
    return L2ReconstructionLoss()


def test_l2_loss_forward(l2_loss: L2ReconstructionLoss) -> None:
    """Test the forward method of L2ReconstructionLoss."""
    input_activations = torch.tensor([[5.0, 4.0], [3.0, 4.0]])
    output_activations = torch.tensor([[1.0, 5.0], [1.0, 5.0]])
    learned_activations = torch.zeros_like(input_activations)

    expected_loss = torch.tensor([8.5, 2.5])
    calculated_loss = l2_loss.forward(input_activations, learned_activations, output_activations)

    assert torch.allclose(calculated_loss, expected_loss), "L2 loss calculation is incorrect."


def test_l2_loss_with_zero_input(l2_loss: L2ReconstructionLoss) -> None:
    """Test the L2 loss function with zero inputs."""
    input_activations = torch.zeros((2, 3))
    output_activations = torch.zeros_like(input_activations)
    learned_activations = torch.zeros_like(input_activations)

    expected_loss = torch.zeros(2)
    calculated_loss = l2_loss.forward(input_activations, learned_activations, output_activations)

    assert torch.all(
        calculated_loss == expected_loss
    ), "L2 loss should be zero for identical zero inputs."


@pytest.mark.parametrize(
    ("input_shape", "learned_activations_shape"),
    [
        ((2, 3), (2, 6)),
        ((2, 4, 3), (2, 4, 6)),
    ],
)
def test_l2_loss_with_varying_input_shapes(
    l2_loss: L2ReconstructionLoss,
    input_shape: tuple[int],
    learned_activations_shape: tuple[int],
    snapshot: SnapshotSession,
) -> None:
    """Test the L2 loss function with varying input shapes."""
    torch.manual_seed(0)
    input_activations = torch.rand(shape_with_optional_dimensions(*input_shape))
    output_activations = torch.rand(shape_with_optional_dimensions(*input_shape))
    learned_activations = torch.rand(shape_with_optional_dimensions(*learned_activations_shape))

    calculated_loss = l2_loss.forward(input_activations, learned_activations, output_activations)

    rounded_zd_loss = torch.round(calculated_loss * 1e5).to(dtype=torch.int).tolist()
    assert str(rounded_zd_loss) == snapshot, "L2 loss has changed from the snapshot."


def test_l2_same_no_components_vs_1_component(l2_loss: L2ReconstructionLoss) -> None:
    """Test the L2 loss is the same with no components vs 1."""
    torch.manual_seed(0)
    input_activations = torch.rand(10, 3)
    learned_activations = torch.rand(10, 5)
    output_activations = torch.rand(10, 3)

    calculated_loss = l2_loss.forward(input_activations, learned_activations, output_activations)
    calculated_loss_1_component = l2_loss.forward(
        input_activations.unsqueeze(1),
        learned_activations.unsqueeze(1),
        output_activations.unsqueeze(1),
    )

    assert torch.allclose(
        calculated_loss, calculated_loss_1_component.squeeze(1)
    ), "L2 loss should be the same with no components vs 1."
