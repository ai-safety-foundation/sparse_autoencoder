"""Abstract generate metric."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
)


@dataclass
class GenerateMetricData:
    """Generate metric data."""

    generated_activations: InputOutputActivationBatch


class AbstractGenerateMetric(ABC):
    """Abstract generate metric."""

    @abstractmethod
    def calculate(self, data: GenerateMetricData) -> dict[str, Any]:
        """Calculate any metrics."""
