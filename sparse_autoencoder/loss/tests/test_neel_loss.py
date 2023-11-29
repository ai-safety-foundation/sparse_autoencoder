"""Tests against Neel's Autoencoder Loss.

Compare module output against Neel's implementation at
https://github.com/neelnanda-io/1L-Sparse-Autoencoder/blob/main/utils.py .
"""
from typing import TypedDict

import pytest
import torch

from sparse_autoencoder.loss.decoded_activations_l2 import L2ReconstructionLoss
from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
from sparse_autoencoder.loss.reducer import LossReducer
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    ItemTensor,
    LearnedActivationBatch,
)


def neel_loss(
    source_activations: InputOutputActivationBatch,
    learned_activations: LearnedActivationBatch,
    decoded_activations: InputOutputActivationBatch,
    l1_coefficient: float,
) -> tuple[ItemTensor, ItemTensor, ItemTensor]:
    """Neel's loss function."""
    l2_loss = (decoded_activations.float() - source_activations.float()).pow(2).sum(-1).mean(0)
    l1_loss = l1_coefficient * (learned_activations.float().abs().sum())
    loss = l2_loss + l1_loss
    return l1_loss, l2_loss, loss


def lib_loss(
    source_activations: InputOutputActivationBatch,
    learned_activations: LearnedActivationBatch,
    decoded_activations: InputOutputActivationBatch,
    l1_coefficient: float,
) -> tuple[ItemTensor, ItemTensor, ItemTensor]:
    """This library's loss function."""
    l1_loss_fn = LearnedActivationsL1Loss(
        l1_coefficient=float(l1_coefficient),
    )
    l2_loss_fn = L2ReconstructionLoss()

    loss_fn = LossReducer(l1_loss_fn, l2_loss_fn)

    l1_loss = l1_loss_fn.forward(source_activations, learned_activations, decoded_activations)
    l2_loss = l2_loss_fn.forward(source_activations, learned_activations, decoded_activations)
    total_loss = loss_fn.forward(source_activations, learned_activations, decoded_activations)
    return l1_loss.sum(), l2_loss.sum(), total_loss.sum()


class MockActivations(TypedDict):
    """Mock activations."""

    source_activations: InputOutputActivationBatch
    learned_activations: LearnedActivationBatch
    decoded_activations: InputOutputActivationBatch


@pytest.fixture()
def mock_activations() -> MockActivations:
    """Create mock activations.

    Returns:
        Tuple of source activations, learned activations, and decoded activations.
    """
    source_activations = torch.rand(10, 20)
    learned_activations = torch.rand(10, 50)
    decoded_activations = torch.rand(10, 20)
    return {
        "source_activations": source_activations,
        "learned_activations": learned_activations,
        "decoded_activations": decoded_activations,
    }


def test_l1_loss_the_same(mock_activations: MockActivations) -> None:
    """Test that the L1 loss is the same."""
    l1_coefficient: float = 0.01

    neel_l1_loss = neel_loss(
        source_activations=mock_activations["source_activations"],
        learned_activations=mock_activations["learned_activations"],
        decoded_activations=mock_activations["decoded_activations"],
        l1_coefficient=l1_coefficient,
    )[0]

    lib_l1_loss = lib_loss(
        source_activations=mock_activations["source_activations"],
        learned_activations=mock_activations["learned_activations"],
        decoded_activations=mock_activations["decoded_activations"],
        l1_coefficient=l1_coefficient,
    )[0].sum()

    assert torch.allclose(neel_l1_loss, lib_l1_loss)


def test_l2_loss_the_same(mock_activations: MockActivations) -> None:
    """Test that the L2 loss is the same."""
    l1_coefficient: float = 0.01

    neel_l2_loss = neel_loss(
        source_activations=mock_activations["source_activations"],
        learned_activations=mock_activations["learned_activations"],
        decoded_activations=mock_activations["decoded_activations"],
        l1_coefficient=l1_coefficient,
    )[1]

    lib_l2_loss = lib_loss(
        source_activations=mock_activations["source_activations"],
        learned_activations=mock_activations["learned_activations"],
        decoded_activations=mock_activations["decoded_activations"],
        l1_coefficient=l1_coefficient,
    )[1].sum()

    # Fix for the fact that Neel's L2 loss is summed across the features dimension and then averaged
    # across the batch. By contrast for l1 it is summed across both features and batch dimensions.
    neel_l2_loss_fixed = neel_l2_loss * len(mock_activations["source_activations"])

    assert torch.allclose(neel_l2_loss_fixed, lib_l2_loss)


@pytest.mark.skip("We believe Neel's L2 approach is different to the original paper.")
def test_total_loss_the_same(mock_activations: MockActivations) -> None:
    """Test that the total loss is the same."""
    l1_coefficient: float = 0.01

    neel_total_loss = neel_loss(
        source_activations=mock_activations["source_activations"],
        learned_activations=mock_activations["learned_activations"],
        decoded_activations=mock_activations["decoded_activations"],
        l1_coefficient=l1_coefficient,
    )[2].sum()

    lib_total_loss = lib_loss(
        source_activations=mock_activations["source_activations"],
        learned_activations=mock_activations["learned_activations"],
        decoded_activations=mock_activations["decoded_activations"],
        l1_coefficient=l1_coefficient,
    )[2].sum()

    assert torch.allclose(neel_total_loss, lib_total_loss)
