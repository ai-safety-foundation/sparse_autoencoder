"""Abstract loss."""
from abc import ABC, abstractmethod
from typing import TypeAlias, final

from strenum import LowercaseStrEnum
import torch
from torch.nn import Module

from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    ItemTensor,
    LearnedActivationBatch,
    TrainBatchStatistic,
)


class LossReductionType(LowercaseStrEnum):
    """Loss reduction type (across batch items)."""

    MEAN = "mean"
    """Mean loss across batch items."""

    SUM = "sum"
    """Sum the loss from all batch items."""


LossLogType: TypeAlias = dict[str, int | float | str]
"""Loss log dict."""


class AbstractLoss(Module, ABC):
    """Abstract loss interface.

    Interface for implementing batch itemwise loss functions.
    """

    _modules: dict[str, "AbstractLoss"]  # type: ignore[assignment] (narrowing)
    """Children loss modules."""

    @abstractmethod
    def forward(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
    ) -> TrainBatchStatistic:
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
    def batch_scalar_loss(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
        reduction: LossReductionType = LossReductionType.MEAN,
    ) -> ItemTensor:
        """Batch scalar loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.
            reduction: Loss reduction type. Typically you would choose LossReductionType.MEAN to
                make the loss independent of the batch size.

        Returns:
            Loss for the batch.
        """
        itemwise_loss = self.forward(source_activations, learned_activations, decoded_activations)

        match reduction:
            case LossReductionType.MEAN:
                return itemwise_loss.mean().squeeze()
            case LossReductionType.SUM:
                return itemwise_loss.sum().squeeze()

    @final
    def batch_scalar_loss_with_log(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
        reduction: LossReductionType = LossReductionType.MEAN,
    ) -> tuple[ItemTensor, LossLogType]:
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
        children_loss_scalars: list[ItemTensor] = []
        metrics: LossLogType = {}

        # If the loss module has children (e.g. it is a reducer):
        if len(self._modules) > 0:
            for loss_module in self._modules.values():
                child_loss, child_metrics = loss_module.batch_scalar_loss_with_log(
                    source_activations,
                    learned_activations,
                    decoded_activations,
                    reduction=reduction,
                )
                children_loss_scalars.append(child_loss)
                metrics.update(child_metrics)

            # Get the total loss & metric
            current_module_loss = torch.stack(children_loss_scalars).sum()

        # Otherwise if it is a leaf loss module:
        else:
            current_module_loss = self.batch_scalar_loss(
                source_activations, learned_activations, decoded_activations, reduction
            )

        # Add in the current loss module's metric
        class_name = self.__class__.__name__
        metrics[class_name] = current_module_loss.detach().cpu().item()

        return current_module_loss, metrics

    @final
    def __call__(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
        reduction: LossReductionType = LossReductionType.MEAN,
    ) -> tuple[ItemTensor, LossLogType]:
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
        return self.batch_scalar_loss_with_log(
            source_activations, learned_activations, decoded_activations, reduction
        )
