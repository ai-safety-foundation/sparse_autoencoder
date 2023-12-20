"""Abstract metric classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.metrics.abstract_metric import (
    AbstractMetric,
    MetricLocation,
    MetricResult,
)
from sparse_autoencoder.metrics.utils.add_component_axis_if_missing import (
    add_component_axis_if_missing,
)
from sparse_autoencoder.tensor_types import Axis


@final
@dataclass
class ValidationMetricData:
    """Validation metric data.

    Dataclass that always has a component axis.
    """

    source_model_loss: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT)]
    """Source model loss (without the SAE)."""

    source_model_loss_with_reconstruction: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT)]
    """Source model loss with SAE reconstruction."""

    source_model_loss_with_zero_ablation: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT)]
    """Source model loss with zero ablation."""

    def __init__(
        self,
        source_model_loss: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)],
        source_model_loss_with_reconstruction: Float[
            Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)
        ],
        source_model_loss_with_zero_ablation: Float[
            Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT_OPTIONAL)
        ],
    ) -> None:
        """Initialize the validation metric data."""
        self.source_model_loss = add_component_axis_if_missing(source_model_loss).detach()
        self.source_model_loss_with_reconstruction = add_component_axis_if_missing(
            source_model_loss_with_reconstruction
        ).detach()
        self.source_model_loss_with_zero_ablation = add_component_axis_if_missing(
            source_model_loss_with_zero_ablation
        ).detach()


class AbstractValidationMetric(AbstractMetric, ABC):
    """Abstract validation metric."""

    @final
    @property
    def location(self) -> MetricLocation:
        """Metric type name."""
        return MetricLocation.VALIDATE

    @abstractmethod
    def calculate(self, data: ValidationMetricData) -> list[MetricResult]:
        """Calculate any metrics."""
