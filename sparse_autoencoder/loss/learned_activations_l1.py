"""Learned activations L1 (absolute error) loss."""
from typing import final

import torch

from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
    TrainBatchStatistic,
)


@final
class LearnedActivationsL1Loss(AbstractLoss):
    """Learned activations L1 (absolute error) loss.

    L1 loss penalty is the absolute sum of the learned activations. The L1 penalty is this
    multiplied by the l1_coefficient (designed to encourage sparsity).

    Example:
        >>> l1_loss = LearnedActivationsL1Loss(0.1)
        >>> learned_activations = torch.tensor([[2.0, -3], [2.0, -3]])
        >>> unused_activations = torch.zeros_like(learned_activations)
        >>> # Returns loss and metrics to log
        >>> l1_loss(unused_activations, learned_activations, unused_activations)
        (tensor(0.5000), {'LearnedActivationsL1Loss': 0.5})
    """

    l1_coefficient: float
    """L1 coefficient."""

    def __init__(self, l1_coefficient: float) -> None:
        """Initialize the absolute error loss.

        Args:
            l1_coefficient: L1 coefficient. The original paper experimented with L1 coefficients of
                [0.01, 0.008, 0.006, 0.004, 0.001]. They used 250 tokens per prompt, so as an
                approximate guide if you use e.g. 2x this number of tokens you might consider using
                0.5x the l1 coefficient.
        """
        self.l1_coefficient = l1_coefficient
        super().__init__()

    def forward(
        self,
        source_activations: InputOutputActivationBatch,  # noqa: ARG002
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,  # noqa: ARG002
    ) -> TrainBatchStatistic:
        """Learned activations L1 (absolute error) loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Loss per batch item.
        """
        absolute_loss = torch.abs(learned_activations)

        return absolute_loss.sum(dim=-1) * self.l1_coefficient

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"l1_coefficient={self.l1_coefficient}"
