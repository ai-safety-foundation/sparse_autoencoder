"""Abstract loss."""
from abc import ABC, abstractmethod
from typing import NamedTuple, final

from jaxtyping import Float
from strenum import LowercaseStrEnum
import torch
from torch import Tensor
from torch.nn import Module

from sparse_autoencoder.metrics.abstract_metric import MetricLocation, MetricResult
from sparse_autoencoder.tensor_types import Axis


class LossReductionType(LowercaseStrEnum):
    """Loss reduction type."""

    MEAN = "mean"

    SUM = "sum"

    NONE = "none"


class LossResultWithMetrics(NamedTuple):
    """Loss result with any metrics to log."""

    loss: Float[Tensor, Axis.COMPONENT] | Float[Tensor, Axis.SINGLE_ITEM]

    loss_metrics: list[MetricResult]


class AbstractLoss(Module, ABC):
    """Abstract loss interface.

    Interface for implementing batch itemwise loss functions.
    """

    _modules: dict[str, "AbstractLoss"]  # type: ignore[assignment] (narrowing)
    """Children loss modules."""

    @abstractmethod
    def log_name(self) -> str:
        """Log name.

        Returns:
            Name of the loss module for logging.
        """

    @abstractmethod
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
        """Batch itemwise loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Loss per batch item.
        """

    @final
    def batch_loss(
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
    ) -> Float[Tensor, Axis.COMPONENT_OPTIONAL]:
        """Batch loss (reduced across the batch axis).

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.
            batch_reduction: Loss reduction type. Typically you would choose LossReductionType.MEAN
                to make the loss independent of the batch size.

        Returns:
            Loss for the batch.

        Raises:
            ValueError: If the batch reduction type is NONE.
        """
        itemwise_loss = self.forward(source_activations, learned_activations, decoded_activations)

        # Reduction parameter is over the batch dimension (not the component dimension)
        match batch_reduction:
            case LossReductionType.MEAN:
                return itemwise_loss.mean(dim=0)
            case LossReductionType.SUM:
                return itemwise_loss.sum(dim=0)
            case LossReductionType.NONE:
                error_message = "Batch reduction type NONE not supported."
                raise ValueError(error_message)

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
        """Scalar loss (reduced across the batch and component axis) with logging.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.
            batch_reduction: Batch reduction type. Typically you would choose LossReductionType.MEAN
                to make the loss independent of the batch size.
            component_reduction: Component reduction type.

        Returns:
            Tuple of the batch scalar loss and a dict of any properties to log.
        """
        children_loss_scalars: list[
            Float[Tensor, Axis.COMPONENT] | Float[Tensor, Axis.SINGLE_ITEM]
        ] = []
        metrics: list[MetricResult] = []

        # If the loss module has children (e.g. it is a reducer):
        if len(self._modules) > 0:
            for loss_module in self._modules.values():
                child_loss, child_metrics = loss_module.scalar_loss_with_log(
                    source_activations,
                    learned_activations,
                    decoded_activations,
                    batch_reduction=batch_reduction,
                    # Note we don't pass through component reduction, as that would prevent logging
                    # component-wise losses in reducers.
                )
                children_loss_scalars.append(child_loss)
                metrics.extend(child_metrics)

            # Get the total loss & metric
            current_module_loss = torch.stack(children_loss_scalars).sum(0)

        # Otherwise if it is a leaf loss module:
        else:
            current_module_loss = self.batch_loss(
                source_activations, learned_activations, decoded_activations, batch_reduction
            )
        # Add in the current loss module's metric
        log = MetricResult(
            location=MetricLocation.TRAIN,
            name="loss",
            postfix=self.log_name(),
            component_wise_values=current_module_loss.unsqueeze(0)
            if current_module_loss.ndim == 0
            else current_module_loss,
        )
        metrics.append(log)

        # Reduce the current module loss across the component dimension
        match component_reduction:
            case LossReductionType.MEAN:
                current_module_loss = current_module_loss.mean(0)
            case LossReductionType.SUM:
                current_module_loss = current_module_loss.sum(0)
            case LossReductionType.NONE:
                pass

        return LossResultWithMetrics(loss=current_module_loss, loss_metrics=metrics)

    @final
    def __call__(
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
        reduction: LossReductionType = LossReductionType.MEAN,
    ) -> LossResultWithMetrics:
        """Batch scalar loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.
            reduction: Loss reduction type. Typically you would choose LossReductionType.MEAN to
                make the loss independent of the batch size.

        Returns:
            Tuple of the batch scalar loss and a dict of any properties to log.
        """
        return self.scalar_loss_with_log(
            source_activations, learned_activations, decoded_activations, reduction
        )
