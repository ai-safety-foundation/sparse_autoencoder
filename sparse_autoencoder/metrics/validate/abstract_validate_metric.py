"""Abstract metric classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.metrics.abstract_metric import (
    AbstractMetric,
    MetricInputData,
    MetricLocation,
    MetricResult,
)
from sparse_autoencoder.tensor_types import Axis


@dataclass
class ValidationMetricData(MetricInputData):
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
        self.source_model_loss = self.add_component_axis_if_missing(source_model_loss)
        self.source_model_loss_with_reconstruction = self.add_component_axis_if_missing(
            source_model_loss_with_reconstruction
        )
        self.source_model_loss_with_zero_ablation = self.add_component_axis_if_missing(
            source_model_loss_with_zero_ablation
        )


class AbstractValidationMetric(AbstractMetric, ABC):
    """Abstract validation metric."""

    @final
    @property
    def metric_location(self) -> MetricLocation:
        """Metric type name."""
        return MetricLocation.VALIDATE

    @abstractmethod
    def calculate(self, data: ValidationMetricData) -> list[MetricResult]:
        """Calculate any metrics."""

    def __init__(self, component_names: list[str] | None = None) -> None:
        """Initialise the metric.

        Args:
            component_names: Component names if there are multiple components.
        """
        super().__init__(component_names=component_names)
