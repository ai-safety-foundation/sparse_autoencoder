"""Tests for the AbstractLoss class."""
import pytest
import torch

from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
    TrainBatchStatistic,
)


class DummyLoss(AbstractLoss):
    """Dummy loss for testing."""

    def forward(
        self,
        source_activations: InputOutputActivationBatch,  # noqa: ARG002
        learned_activations: LearnedActivationBatch,  # noqa: ARG002
        decoded_activations: InputOutputActivationBatch,  # noqa: ARG002
    ) -> TrainBatchStatistic:
        """Batch itemwise loss."""
        # Simple dummy implementation for testing
        return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture()
def dummy_loss() -> DummyLoss:
    """Dummy loss for testing."""
    return DummyLoss()


def test_abstract_class_enforced() -> None:
    """Test that initializing the abstract class raises an error."""
    with pytest.raises(TypeError):
        AbstractLoss()  # type: ignore


@pytest.mark.parametrize(
    ("loss_reduction", "expected"),
    [
        (LossReductionType.MEAN, 2.0),  # Mean of [1.0, 2.0, 3.0]
        (LossReductionType.SUM, 6.0),  # Sum of [1.0, 2.0, 3.0]
    ],
)
def test_batch_scalar_loss(
    dummy_loss: DummyLoss, loss_reduction: LossReductionType, expected: float
) -> None:
    """Test the batch scalar loss."""
    source_activations = learned_activations = decoded_activations = torch.ones((1, 3))

    loss_mean = dummy_loss.batch_scalar_loss(
        source_activations, learned_activations, decoded_activations, loss_reduction
    )
    assert loss_mean.item() == expected


def test_batch_scalar_loss_with_log(dummy_loss: DummyLoss) -> None:
    """Test the batch scalar loss with log."""
    source_activations = learned_activations = decoded_activations = torch.ones((1, 3))
    _loss, log = dummy_loss.batch_scalar_loss_with_log(
        source_activations, learned_activations, decoded_activations
    )
    assert "DummyLoss" in log
    expected = 2.0  # Mean of [1.0, 2.0, 3.0]
    assert log["DummyLoss"] == expected


def test_call_method(dummy_loss: DummyLoss) -> None:
    """Test the call method."""
    source_activations = learned_activations = decoded_activations = torch.ones((1, 3))
    _loss, log = dummy_loss(source_activations, learned_activations, decoded_activations)
    assert "DummyLoss" in log
    expected = 2.0  # Mean of [1.0, 2.0, 3.0]
    assert log["DummyLoss"] == expected
