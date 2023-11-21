"""Abstract metric classes."""
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.metrics.abstract_metric import AbstractMetric


@dataclass
class ValidationMetricData:
    """Validation metric data."""

    source_model_loss: float

    autoencoder_loss: float


class AbstractValidationMetric(AbstractMetric):
    """Abstract validation metric."""

    @abstractmethod
    def create_progress_bar_postfix(self, data: ValidationMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @abstractmethod
    def create_weights_and_biases_log(self, data: ValidationMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        raise NotImplementedError
