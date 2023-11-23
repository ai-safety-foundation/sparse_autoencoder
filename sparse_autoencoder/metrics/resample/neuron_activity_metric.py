"""Neuron activity metric."""
from typing import Any

import pandas as pd
import wandb

from sparse_autoencoder.metrics.resample.abstract_resample_metric import (
    AbstractResampleMetric,
    ResampleMetricData,
)


class NeuronActivityMetric(AbstractResampleMetric):
    """Neuron activity metric."""

    def calculate(self, data: ResampleMetricData) -> dict[str, Any]:
        """Calculate the neuron activity metrics.

        Args:
            data: Resample metric data.

        Returns:
            Dictionary of metrics.
        """
        neuron_activity = data.neuron_activity
        neuron_activity_list = neuron_activity.detach().cpu().tolist()

        # Histogram of neuron activity
        histogram = wandb.Histogram(neuron_activity_list)

        return {
            "resample_alive_neuron_count": (neuron_activity > 0).sum().item(),
            "resample_dead_neuron_count": (neuron_activity == 0).sum().item(),
            "resample_neuron_activity_histogram": histogram,
        }
