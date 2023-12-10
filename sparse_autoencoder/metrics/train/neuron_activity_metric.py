"""Neuron activity metric.

Logs the number of dead and alive neurons at various horizons. Also logs histograms of neuron
activity, and the number of neurons that are almost dead.
"""
from typing import Any

from jaxtyping import Int64
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import wandb

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)
from sparse_autoencoder.tensor_types import Axis


DEFAULT_HORIZONS = [10_000, 100_000, 500_000, 1_000_000, 10_000_000]
"""Default horizons."""

DEFAULT_THRESHOLDS = [1e-5, 1e-6]
"""Default thresholds for determining if a neuron is almost dead."""


class NeuronActivityHorizonData:
    """Neuron activity data for a single horizon.

    For each time horizon we store some data (e.g. the number of times each neuron fired inside this
    time horizon). This class also contains some helper methods for then calculating metrics from
    this data.
    """

    _horizon_number_activations: int
    """Horizon in number of activations."""

    _horizon_steps: int
    """Horizon in number of steps."""

    _steps_since_last_calculated: int
    """Steps since last calculated."""

    _neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE]
    """Neuron activity since inception."""

    _thresholds: list[float]
    """Thresholds for almost dead neurons."""

    @property
    def _dead_count(self) -> int:
        """Dead count."""
        dead_bool_mask: Int64[Tensor, Axis.LEARNT_FEATURE] = self._neuron_activity == 0
        count_dead: Int64[Tensor, Axis.SINGLE_ITEM] = dead_bool_mask.sum()
        return int(count_dead.item())

    @property
    def _dead_fraction(self) -> float:
        """Dead fraction."""
        return self._dead_count / self._neuron_activity.shape[-1]

    @property
    def _alive_count(self) -> int:
        """Alive count."""
        alive_bool_mask: Int64[Tensor, Axis.LEARNT_FEATURE] = self._neuron_activity > 0
        count_alive: Int64[Tensor, Axis.SINGLE_ITEM] = alive_bool_mask.sum()
        return int(count_alive.item())

    def _almost_dead(self, threshold: float) -> int | None:
        """Almost dead count."""
        threshold_in_activations: float = threshold * self._horizon_number_activations
        if threshold_in_activations < 1:
            return None

        almost_dead_bool_mask: Int64[Tensor, Axis.LEARNT_FEATURE] = (
            self._neuron_activity < threshold_in_activations
        )
        count_almost_dead: Int64[Tensor, Axis.SINGLE_ITEM] = almost_dead_bool_mask.sum()
        return int(count_almost_dead.item())

    @property
    def _activity_histogram(self) -> wandb.Histogram:
        """Activity histogram."""
        numpy_neuron_activity: NDArray[np.float_] = self._neuron_activity.detach().cpu().numpy()
        bins, values = np.histogram(numpy_neuron_activity, bins=50)
        return wandb.Histogram(np_histogram=(bins, values))

    @property
    def _log_activity_histogram(self) -> wandb.Histogram:
        """Log activity histogram."""
        numpy_neuron_activity: NDArray[np.float_] = self._neuron_activity.detach().cpu().numpy()
        log_epsilon = 0.1  # To avoid log(0)
        log_neuron_activity = np.log(numpy_neuron_activity + log_epsilon)
        bins, values = np.histogram(log_neuron_activity, bins=50)
        return wandb.Histogram(np_histogram=(bins, values))

    @property
    def name(self) -> str:
        """Name."""
        return f"over_{self._horizon_number_activations}_activations"

    @property
    def wandb_log_values(self) -> dict[str, Any]:
        """Wandb log values."""
        log = {
            f"train/activity/{self.name}/dead_count": self._dead_count,
            f"train/activity/{self.name}/alive_count": self._alive_count,
            f"train/activity/{self.name}/activity_histogram": self._activity_histogram,
            f"train/activity/{self.name}/log_activity_histogram": self._log_activity_histogram,
        }

        for threshold in self._thresholds:
            almost_dead_count = self._almost_dead(threshold)
            if almost_dead_count is not None:
                log[f"train/activity/{self.name}/almost_dead_{threshold}"] = almost_dead_count

        return log

    def __init__(
        self,
        approximate_activation_horizon: int,
        train_batch_size: int,
        number_learned_features: int,
        thresholds: list[float],
    ) -> None:
        """Initialise the neuron activity horizon data.

        Args:
            approximate_activation_horizon: Approximate activation horizon.
            train_batch_size: Train batch size.
            number_learned_features: Number of learned features.
            thresholds: Thresholds for almost dead neurons.
        """
        self._steps_since_last_calculated = 0
        self._neuron_activity = torch.zeros(number_learned_features, dtype=torch.int64)
        self._thresholds = thresholds

        # Get a precise activation_horizon
        self._horizon_steps = approximate_activation_horizon // train_batch_size
        self._horizon_number_activations = self._horizon_steps * train_batch_size

    def step(self, neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE]) -> dict[str, Any]:
        """Step the neuron activity horizon data.

        Args:
            neuron_activity: Neuron activity.

        Returns:
            Dictionary of metrics (or empty dictionary if no metrics are ready to be logged).
        """
        self._steps_since_last_calculated += 1
        self._neuron_activity += neuron_activity

        if self._steps_since_last_calculated >= self._horizon_steps:
            result = {**self.wandb_log_values}
            self._steps_since_last_calculated = 0
            self._neuron_activity = torch.zeros_like(self._neuron_activity)
            return result

        return {}


class NeuronActivityMetric(AbstractTrainMetric):
    """Neuron activity metric."""

    _approximate_horizons: list[int]

    _data: list[NeuronActivityHorizonData]

    _initialised: bool = False

    _thresholds: list[float]

    def __init__(
        self,
        approximate_horizons: list[int] = DEFAULT_HORIZONS,
        thresholds: list[float] = DEFAULT_THRESHOLDS,
    ) -> None:
        """Initialise the neuron activity metric.

        time `calculate` is called.

        Args:
            approximate_horizons: Approximate horizons in number of activations.
            thresholds: Thresholds for almost dead neurons.
        """
        super().__init__()
        self._approximate_horizons = approximate_horizons
        self._data = []
        self._thresholds = thresholds

    def initialise_horizons(self, data: TrainMetricData) -> None:
        """Initialise the horizon data structures.

        Args:
            data: Train metric data.
        """
        train_batch_size = data.learned_activations.shape[0]
        number_learned_features = data.learned_activations.shape[-1]

        for horizon in self._approximate_horizons:
            # Don't add horizons that are smaller than the train batch size
            if horizon < train_batch_size:
                continue

            self._data.append(
                NeuronActivityHorizonData(
                    approximate_activation_horizon=horizon,
                    train_batch_size=train_batch_size,
                    number_learned_features=number_learned_features,
                    thresholds=self._thresholds,
                )
            )

        self._initialised = True

    def calculate(self, data: TrainMetricData) -> dict[str, Any]:
        """Calculate the neuron activity metrics.

        Args:
            data: Resample metric data.

        Returns:
            Dictionary of metrics.
        """
        if not self._initialised:
            self.initialise_horizons(data)

        log = {}

        for horizon_data in self._data:
            fired_count: Int64[Tensor, Axis.LEARNT_FEATURE] = (
                (data.learned_activations > 0).sum(dim=0).detach().cpu()
            )
            horizon_specific_log = horizon_data.step(fired_count)
            log.update(horizon_specific_log)

        return log
