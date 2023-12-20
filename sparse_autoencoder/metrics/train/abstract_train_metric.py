"""Abstract train metric."""
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
class TrainMetricData:
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
        self.input_activations = add_component_axis_if_missing(
            input_activations, dimensions_without_component=2
        ).detach()
        self.learned_activations = add_component_axis_if_missing(
            learned_activations, dimensions_without_component=2
        ).detach()
        self.decoded_activations = add_component_axis_if_missing(
            decoded_activations, dimensions_without_component=2
        ).detach()


class AbstractTrainMetric(AbstractMetric, ABC):
    """Abstract train metric."""

    @final
    @property
    def location(self) -> MetricLocation:
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
