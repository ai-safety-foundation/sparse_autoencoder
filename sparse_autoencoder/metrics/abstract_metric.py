"""Abstract metric."""
from abc import ABC
from typing import final


class AbstractMetric(ABC):
    """Abstract metric."""

    _should_log_progress_bar: bool

    _should_log_weights_and_biases: bool

    @final
    def __init__(self, *, log_progress_bar: bool = False, log_weights_and_biases: bool = True):
        """Initialise the train metric."""
        self._should_log_progress_bar = log_progress_bar
        self._should_log_weights_and_biases = log_weights_and_biases
