"""Abstract train metric."""
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
class TrainMetricData(MetricInputData):
    """Train metric data."""

    input_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Input activations."""

    learned_activations: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    """Learned activations."""

    decoded_activations: Float[
        Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Decoded activations."""

    def __init__(
        self,
        input_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Initialize the train metric data."""
        self.input_activations = self.add_component_axis_if_missing(
            input_activations, dimensions_without_component=2
        )
        self.learned_activations = self.add_component_axis_if_missing(
            learned_activations, dimensions_without_component=2
        )
        self.decoded_activations = self.add_component_axis_if_missing(
            decoded_activations, dimensions_without_component=2
        )


class AbstractTrainMetric(AbstractMetric, ABC):
    """Abstract train metric."""

    @final
    @property
    def metric_location(self) -> MetricLocation:
        """Metric type name."""
        return MetricLocation.TRAIN

    @abstractmethod
    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Calculate any metrics component wise.

        Args:
            data: Train metric data.

        Returns:
            Dictionary of metrics.
        """

    def __init__(self, component_names: list[str] | None = None) -> None:
        """Initialise the metric.

        Args:
            component_names: Component names if there are multiple components.
        """
        super().__init__(component_names=component_names)
