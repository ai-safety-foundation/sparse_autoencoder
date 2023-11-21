"""Abstract train metric."""
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sparse_autoencoder.metrics.abstract_metric import AbstractMetric
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


class AbstractTrainMetric(AbstractMetric):
    """Abstract train metric."""

    @abstractmethod
    def create_progress_bar_postfix(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @abstractmethod
    def create_weights_and_biases_log(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        raise NotImplementedError
