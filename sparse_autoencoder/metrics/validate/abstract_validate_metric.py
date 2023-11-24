"""Abstract metric classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from transformer_lens import HookedTransformer

from sparse_autoencoder.autoencoder.abstract_autoencoder import AbstractAutoencoder
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset


@dataclass
class ValidationMetricContext:
    """Validation metric data."""

    autoencoder: AbstractAutoencoder
    source_model: HookedTransformer
    dataset: SourceDataset
    hook_point: str


class AbstractValidationMetric(ABC):
    """Abstract validation metric."""

    @abstractmethod
    def calculate(self, context: ValidationMetricContext) -> dict[str, Any]:
        """Calculate any metrics."""
