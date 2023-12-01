"""Neuron activity metric."""
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
import wandb

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import Axis
from jaxtyping import Int
from torch import Tensor


class NeuronActivityMetric(AbstractTrainMetric):
    """Neuron activity metric."""

    def __init__(self, horizon: int = 100) -> None:
        """Initialise the neuron activity metric.

        Args:
            horizon: Number of batches to average over.
        """
        super().__init__()
        self.horizon = horizon
        self.neuron_activity: None | Int[Tensor, Axis.LEARNT_FEATURE] = None
        self.batch_count = 0
        self.activation_eps = 0.1

    def calculate(self, data: TrainMetricData) -> dict[str, Any]:
        """Calculate the neuron activity metrics.

        Args:
            data: Resample metric data.

        Returns:
            Dictionary of metrics.
        """
        fired = data.learned_activations > 0
        if self.batch_count == 0:
            self.neuron_activity = fired.sum(dim=0)
        else:
            self.neuron_activity += fired.sum(dim=0)

        self.batch_count += 1

        if self.batch_count >= self.horizon:
            # Histogram of neuron activity
            numpy_neuron_activity: NDArray[np.float_] = self.neuron_activity.detach().cpu().numpy()
            log_neuron_activity = np.log(numpy_neuron_activity + self.activation_eps)
            bins, values = np.histogram(numpy_neuron_activity, bins=50)
            histogram = wandb.Histogram(np_histogram=(bins, values))
            log_bins, log_values = np.histogram(log_neuron_activity, bins=50)
            log_histogram = wandb.Histogram(np_histogram=(log_bins, log_values))

            self.batch_count = 0
            return {
                "alive_neuron_count": (self.neuron_activity > 0).sum().item(),
                "dead_neuron_count": (self.neuron_activity == 0).sum().item(),
                "neuron_activity_histogram": histogram,
                "log_neuron_activity_histogram": log_histogram,
            }
        return {}
