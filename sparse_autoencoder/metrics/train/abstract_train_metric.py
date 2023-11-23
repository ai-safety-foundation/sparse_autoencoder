"""Abstract train metric."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
)


@dataclass
class TrainMetricData:
    """Train metric data."""

    input_activations: InputOutputActivationBatch

    learned_activations: LearnedActivationBatch

    decoded_activations: InputOutputActivationBatch


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
