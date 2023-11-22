"""L0 (sparsity) norm metric."""
from collections import OrderedDict
from typing import Any, final

import wandb

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


@final
class HistogramMetric(AbstractTrainMetric):
    """Histogram metric — log the histogram, as well as the number of dead features."""

    @final
    def __init__(
        self,
        dead_threshold: float = 0.0,
    ) -> None:
        """Create a histogram metric.

        Args:
            dead_threshold: Threshold for considering a feature to be dead.
            log_progress_bar: Whether to log the metric to the progress bar.
            log_weights_and_biases: Whether to log the metric to Weights and Biases.
        """
        super().__init__()
        self.dead_threshold = dead_threshold

    @final
    def calculate(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        feature_mean_acts = data.learned_activations.mean(dim=0)
        num_dead_features = (feature_mean_acts <= self.dead_threshold).sum().item()

        return OrderedDict(
            histogram=wandb.Histogram(feature_mean_acts.tolist()),
            num_dead_features=num_dead_features,
        )
