"""Abstract metric classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


@dataclass
class ValidationMetricData:
    """Validation metric data."""

    source_model_loss: Float[Tensor, Axis.ITEMS]

    source_model_loss_with_reconstruction: Float[Tensor, Axis.ITEMS]

    source_model_loss_with_zero_ablation: Float[Tensor, Axis.ITEMS]


class AbstractValidationMetric(ABC):
    """Abstract validation metric."""

    @abstractmethod
    def calculate(self, data: ValidationMetricData) -> dict[str, Any]:
        """Calculate any metrics."""
