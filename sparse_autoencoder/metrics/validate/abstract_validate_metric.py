"""Abstract metric classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationMetricData:
    """Validation metric data."""

    source_model_loss: float

    autoencoder_loss: float


class AbstractValidationMetric(ABC):
    """Abstract validation metric."""

    @abstractmethod
    def calculate(self, data: ValidationMetricData) -> dict[str, Any]:
        """Calculate any metrics."""
