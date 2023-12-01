"""Abstract train metric."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


@dataclass
class TrainMetricData:
    """Train metric data."""

    input_activations: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]

    learned_activations: Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)]

    decoded_activations: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]


class AbstractTrainMetric(ABC):
    """Abstract train metric."""

    @abstractmethod
    def calculate(self, data: TrainMetricData) -> dict[str, Any]:
        """Calculate any metrics.

        Args:
            data: Train metric data.

        Returns:
            Dictionary of metrics.
        """
