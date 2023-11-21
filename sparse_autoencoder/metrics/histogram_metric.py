"""L0 (sparsity) norm metric."""
from collections import OrderedDict
from typing import Any, final

import wandb

from sparse_autoencoder.metrics.abstract_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


@final
class HistogramMetric(AbstractTrainMetric):
    """Histogram metric — log the histogram, as well as the number of dead features."""

    @final
    def __init__(self,
                 dead_threshold: float = 0.0,
                 log_progress_bar: bool = False,
                 log_weights_and_biases: bool = True):
        """Create a histogram metric.

        Args:
            dead_threshold: Threshold for considering a feature to be dead.
            log_progress_bar: Whether to log the metric to the progress bar.
            log_weights_and_biases: Whether to log the metric to Weights and Biases.
        """
        super().__init__(log_progress_bar=log_progress_bar,
                         log_weights_and_biases=log_weights_and_biases)
        self.dead_threshold = dead_threshold

    """Histogram metric — log the histogram, as well as the number of dead features"""
    @final
    def create_progress_bar_postfix(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a progress bar postfix."""
        raise NotImplementedError

    @final
    def create_weights_and_biases_log(self, data: TrainMetricData) -> OrderedDict[str, Any]:
        """Create a log item for Weights and Biases."""
        feature_mean_acts = data.learned_activations.mean(dim=0)
        num_dead_features = (feature_mean_acts <= self.dead_threshold).sum().item()

        return OrderedDict(
            histogram=wandb.Histogram(feature_mean_acts.tolist()),
            num_dead_features=num_dead_features,
        )
