"""Neuron activity metric."""
from typing import Any

import numpy as np
from numpy.typing import NDArray
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

        # Histogram of neuron activity
        numpy_neuron_activity: NDArray[np.float_] = neuron_activity.detach().cpu().numpy()
        bins, values = np.histogram(numpy_neuron_activity, bins=50)
        histogram = wandb.Histogram(np_histogram=(bins, values))

        return {
            "resample_alive_neuron_count": (neuron_activity > 0).sum().item(),
            "resample_dead_neuron_count": (neuron_activity == 0).sum().item(),
            "resample_neuron_activity_histogram": histogram,
        }
