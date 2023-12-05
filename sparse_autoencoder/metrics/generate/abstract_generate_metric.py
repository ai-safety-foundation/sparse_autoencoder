"""Abstract generate metric."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


@dataclass
class GenerateMetricData:
    """Generate metric data."""

    generated_activations: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]


class AbstractGenerateMetric(ABC):
    """Abstract generate metric."""

    @abstractmethod
    def calculate(self, data: GenerateMetricData) -> dict[str, Any]:
        """Calculate any metrics."""
