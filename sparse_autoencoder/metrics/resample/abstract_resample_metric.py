"""Abstract resample metric."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.tensor_types import (
    NeuronActivity,
)


@dataclass
class ResampleMetricData:
    """Resample metric data."""

    neuron_activity: NeuronActivity
    """Number of times each neuron fired."""


class AbstractResampleMetric(ABC):
    """Abstract resample metric."""

    @abstractmethod
    def calculate(self, data: ResampleMetricData) -> dict[str, Any]:
        """Calculate any metrics.

        Args:
            data: Resample metric data.

        Returns:
            Dictionary of metrics.
        """
