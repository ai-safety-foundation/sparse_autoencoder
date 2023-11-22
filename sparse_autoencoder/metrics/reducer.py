"""Metric reducer."""
from collections import OrderedDict
from collections.abc import Iterator
from typing import final

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


@final
class TrainingMetricReducer(AbstractTrainMetric):
    """Training metric reducer.

    Reduces multiple training metrics into a single training metric (by merging
    their OrderedDicts).
    """

    _modules: list["AbstractTrainMetric"]
    """Children training metric modules."""

    @final
    def __init__(
        self,
        *metric_modules: AbstractTrainMetric,
    ):
        """Initialize the training metric reducer.

        Args:
            metric_modules: Training metric modules to reduce.

        Raises:
            ValueError: If the training metric reducer has no training metric modules.
        """
        super().__init__()

        self._modules = list(metric_modules)

        if len(self) == 0:
            error_message = "Training metric reducer must have at least one training metric module."
            raise ValueError(error_message)

    @final
    def calculate(self, data: TrainMetricData) -> OrderedDict[str, float]:
        """Create a log item for Weights and Biases."""
        result = OrderedDict()
        for module in self._modules:
            result.update(module.calculate(data))
        return result

    def __dir__(self) -> list[str]:
        """Dir dunder method."""
        return list(self._modules.__dir__())

    def __getitem__(self, idx: int) -> AbstractTrainMetric:
        """Get item dunder method."""
        return self._modules[idx]

    def __iter__(self) -> Iterator[AbstractTrainMetric]:
        """Iterator dunder method."""
        return iter(self._modules)

    def __len__(self) -> int:
        """Length dunder method."""
        return len(self._modules)
