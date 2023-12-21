"""Tests for the AbstractLoss class."""
from jaxtyping import Float
import pytest
import torch
from torch import Tensor

from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.tensor_types import Axis


class DummyLoss(AbstractLoss):
    """Dummy loss for testing."""

    def forward(
        self,
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]:
        """Batch itemwise loss."""
        # Default to 1D tensor if no component axis
        ndim_with_no_component_axis = 2
        if source_activations.ndim == ndim_with_no_component_axis:
            return torch.tensor([1.0, 2.0, 3.0])  # Loss for 3 batches

        # If there is a component axis, duplicate the loss for each component
        single_component_loss = torch.tensor([1.0, 2, 3])
        return single_component_loss.repeat(source_activations.shape[1], 1).T

    def log_name(self) -> str:
        """Log name."""
        return "dummy"


@pytest.fixture()
def dummy_loss() -> DummyLoss:
    """Dummy loss for testing."""
    return DummyLoss()


@pytest.mark.parametrize(
    ("n_components", "loss_reduction", "expected"),
    [
        (None, LossReductionType.MEAN, 2.0),  # Mean of [1.0, 2.0, 3.0]
        (None, LossReductionType.SUM, 6.0),  # Sum of [1.0, 2.0, 3.0]
        (1, LossReductionType.MEAN, 2.0),  # Same as above
        (1, LossReductionType.SUM, 6.0),  # Same as above
        (2, LossReductionType.MEAN, 4.0),  # Double
        (2, LossReductionType.SUM, 12.0),  # Double
    ],
)
def test_batch_loss(
    dummy_loss: DummyLoss,
    n_components: int | None,
    loss_reduction: LossReductionType,
    expected: float,
) -> None:
    """Test the batch loss."""
    if n_components is None:
        source_activations = decoded_activations = torch.ones(3, 12)
        learned_activations = torch.ones(3, 24)
    else:
        source_activations = decoded_activations = torch.ones((3, n_components, 12))
        learned_activations = torch.ones((3, n_components, 24))

    batch_loss = dummy_loss.batch_loss(
        source_activations, learned_activations, decoded_activations, loss_reduction
    )

    assert batch_loss.sum() == expected


def test_batch_loss_with_log(dummy_loss: DummyLoss) -> None:
    """Test the scalar loss with log."""
    source_activations = learned_activations = decoded_activations = torch.ones((1, 3))
    _loss, log = dummy_loss.scalar_loss_with_log(
        source_activations, learned_activations, decoded_activations
    )
    expected = 2.0  # Mean of [1.0, 2.0, 3.0]
    assert log[0].component_wise_values[0] == expected


def test_scalar_loss_with_log_and_component_axis(dummy_loss: DummyLoss) -> None:
    """Test the scalar loss with log and component axis."""
    n_components = 3
    source_activations = decoded_activations = torch.ones((3, n_components, 12))
    learned_activations = torch.ones((3, n_components, 24))
    _loss, log = dummy_loss.scalar_loss_with_log(
        source_activations, learned_activations, decoded_activations
    )
    expected = 2.0  # Mean of [1.0, 2.0, 3.0]
    for component_idx in range(n_components):
        assert log[0].component_wise_values[component_idx] == expected


def test_call_method(dummy_loss: DummyLoss) -> None:
    """Test the call method."""
    source_activations = learned_activations = decoded_activations = torch.ones((1, 3))
    _loss, log = dummy_loss(source_activations, learned_activations, decoded_activations)
    expected = 2.0  # Mean of [1.0, 2.0, 3.0]
    assert log[0].component_wise_values[0] == expected
