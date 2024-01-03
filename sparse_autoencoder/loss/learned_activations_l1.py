"""Learned activations L1 (absolute error) loss."""
from typing import NamedTuple, final

from jaxtyping import Float
from pydantic import PositiveFloat, validate_call
import torch
from torch import Tensor

from sparse_autoencoder.loss.abstract_loss import (
    AbstractLoss,
    LossReductionType,
    LossResultWithMetrics,
)
from sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult
from sparse_autoencoder.tensor_types import Axis


class _L1LossAndPenalty(NamedTuple):
    """L1 loss result and loss penalty."""

    itemwise_absolute_loss: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]
    """Itemwise absolute loss."""

    itemwise_absolute_loss_penalty: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]
    """Itemwise absolute loss multiplied by the l1 coefficient."""


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
        >>> l1_loss.forward(unused_activations, learned_activations, unused_activations)[0]
        tensor(0.5000)
    """

    l1_coefficient: float | Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)]
    """L1 coefficient."""

    def log_name(self) -> str:
        """Log name.

        Returns:
            Name of the loss module for logging.
        """
        return "learned_activations_l1_loss_penalty"

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self, l1_coefficient: PositiveFloat | Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)]
    ) -> None:
        """Initialize the absolute error loss.

        Args:
            l1_coefficient: L1 coefficient. The original paper experimented with L1 coefficients of
                [0.01, 0.008, 0.006, 0.004, 0.001]. They used 250 tokens per prompt, so as an
                approximate guide if you use e.g. 2x this number of tokens you might consider using
                0.5x the l1 coefficient.
        """
        self.l1_coefficient = l1_coefficient
        super().__init__()

    def _l1_loss(
        self,
        source_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[  # noqa: ARG002s
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> _L1LossAndPenalty:
        """Learned activations L1 (absolute error) loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Tuple of itemwise absolute loss, and itemwise absolute loss multiplied by the l1
            coefficient.
        """
        # Absolute loss is the summed absolute value of the learned activations (i.e. over the
        # learned feature axis).
        itemwise_absolute_loss: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)
        ] = torch.abs(learned_activations).sum(dim=-1)

        itemwise_absolute_loss_penalty: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)
        ] = itemwise_absolute_loss * self.l1_coefficient

        return _L1LossAndPenalty(
            itemwise_absolute_loss=itemwise_absolute_loss,
            itemwise_absolute_loss_penalty=itemwise_absolute_loss_penalty,
        )

    def forward(
        self,
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]:
        """Learned activations L1 (absolute error) loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Loss per batch item.
        """
        return self._l1_loss(
            source_activations, learned_activations, decoded_activations
        ).itemwise_absolute_loss_penalty

    # Override to add both the loss and the penalty to the log
    def scalar_loss_with_log(
        self,
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        batch_reduction: LossReductionType = LossReductionType.MEAN,
        component_reduction: LossReductionType = LossReductionType.NONE,
    ) -> LossResultWithMetrics:
        """Scalar L1 loss (reduced across the batch and component axis) with logging.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.
            batch_reduction: Batch reduction type. Typically you would choose LossReductionType.MEAN
                to make the loss independent of the batch size.
            component_reduction: Component reduction type.

        Returns:
            Tuple of the L1 absolute error batch scalar loss and a dict of the properties to log
                (loss before and after the l1 coefficient).

        Raises:
            ValueError: If batch_reduction is LossReductionType.NONE.
        """
        itemwise_absolute_loss, itemwise_absolute_loss_penalty = self._l1_loss(
            source_activations, learned_activations, decoded_activations
        )

        match batch_reduction:
            case LossReductionType.MEAN:
                batch_scalar_loss = itemwise_absolute_loss.mean(0)
                batch_scalar_loss_penalty = itemwise_absolute_loss_penalty.mean(0)
            case LossReductionType.SUM:
                batch_scalar_loss = itemwise_absolute_loss.sum(0)
                batch_scalar_loss_penalty = itemwise_absolute_loss_penalty.sum(0)
            case LossReductionType.NONE:
                error_message = "Batch reduction type NONE not supported."
                raise ValueError(error_message)

        # Create the log
        metrics: list[MetricResult] = [
            MetricResult(
                name="loss",
                postfix="learned_activations_l1",
                component_wise_values=batch_scalar_loss.unsqueeze(0)
                if batch_scalar_loss.ndim == 0
                else batch_scalar_loss,
                location=MetricLocation.TRAIN,
            ),
            MetricResult(
                name="loss",
                postfix=self.log_name(),
                component_wise_values=batch_scalar_loss_penalty.unsqueeze(0)
                if batch_scalar_loss_penalty.ndim == 0
                else batch_scalar_loss_penalty,
                location=MetricLocation.TRAIN,
            ),
        ]

        match component_reduction:
            case LossReductionType.MEAN:
                batch_scalar_loss_penalty = batch_scalar_loss_penalty.mean(0)
            case LossReductionType.SUM:
                batch_scalar_loss_penalty = batch_scalar_loss_penalty.sum(0)
            case LossReductionType.NONE:
                pass

        return LossResultWithMetrics(loss=batch_scalar_loss_penalty, loss_metrics=metrics)

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"l1_coefficient={self.l1_coefficient}"
