"""Abstract generate metric."""
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.metrics.abstract_metric import AbstractMetric
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
)


@dataclass
class GenerateMetricData:
    """Generate metric data."""

    generated_activations: InputOutputActivationBatch


class AbstractGenerateMetric(AbstractMetric):
    """Abstract generate metric."""

    @abstractmethod
    def create_progress_bar_postfix(self, data: GenerateMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @abstractmethod
    def create_weights_and_biases_log(self, data: GenerateMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        raise NotImplementedError
