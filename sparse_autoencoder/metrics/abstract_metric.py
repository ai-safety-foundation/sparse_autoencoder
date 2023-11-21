"""Abstract metric classes."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, final

from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
)


@dataclass
class GenerateMetricData:
    """Generate metric data."""

    generated_activations: InputOutputActivationBatch


@dataclass
class TrainMetricData:
    """Train metric data."""

    input_activations: InputOutputActivationBatch

    learned_activations: LearnedActivationBatch

    decoded_activations: InputOutputActivationBatch


@dataclass
class ValidationMetricData:
    """Validation metric data."""

    source_model_loss: float

    autoencoder_loss: float


class AbstractMetric(ABC):
    """Abstract metric."""

    _should_log_progress_bar: bool

    _should_log_weights_and_biases: bool

    @final
    def __init__(self, *, log_progress_bar: bool = False, log_weights_and_biases: bool = True):
        """Initialise the train metric."""
        self._should_log_progress_bar = log_progress_bar
        self._should_log_weights_and_biases = log_weights_and_biases


class AbstractGenerateMetric(AbstractMetric, ABC):
    """Abstract generate metric."""

    @abstractmethod
    def create_progress_bar_postfix(self, data: GenerateMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @abstractmethod
    def create_weights_and_biases_log(self, data: GenerateMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        raise NotImplementedError


class AbstractTrainMetric(AbstractMetric, ABC):
    """Abstract train metric."""

    @abstractmethod
    def create_progress_bar_postfix(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @abstractmethod
    def create_weights_and_biases_log(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        raise NotImplementedError


class AbstractValidationMetric(AbstractMetric, ABC):
    """Abstract validation metric."""

    @abstractmethod
    def create_progress_bar_postfix(self, data: ValidationMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @abstractmethod
    def create_weights_and_biases_log(self, data: ValidationMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        raise NotImplementedError
