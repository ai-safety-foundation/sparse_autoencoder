"""Abstract metric classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.tensor_types import ValidationStatistics


@dataclass
class ValidationMetricData:
    """Validation metric data."""

    source_model_loss: ValidationStatistics

    source_model_loss_with_reconstruction: ValidationStatistics

    source_model_loss_with_zero_ablation: ValidationStatistics


class AbstractValidationMetric(ABC):
    """Abstract validation metric."""

    @abstractmethod
    def calculate(self, data: ValidationMetricData) -> dict[str, Any]:
        """Calculate any metrics."""
