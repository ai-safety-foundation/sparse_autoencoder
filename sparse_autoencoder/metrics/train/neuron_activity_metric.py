"""Neuron activity metric.

Logs the number of dead and alive neurons at various horizons. Also logs histograms of neuron
activity, and the number of neurons that are almost dead.
"""
from jaxtyping import Float, Int, Int64
import numpy as np
import torch
from torch import Tensor
import wandb

from sparse_autoencoder.metrics.abstract_metric import (
    MetricLocation,
    MetricResult,
)
from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
    TrainMetricData,
)
from sparse_autoencoder.tensor_types import Axis


DEFAULT_HORIZONS = [10_000, 100_000, 1_000_000, 10_000_000]
"""Default horizons (in number of logged activations)."""

DEFAULT_THRESHOLDS = [1e-5, 1e-6]
"""Default thresholds for determining if a neuron is almost dead."""


class NeuronActivityHorizonData:
    """Neuron activity data for a specific horizon (number of activations seen).

    For each time horizon we store some data (e.g. the number of times each neuron fired inside this
    time horizon). This class also contains some helper methods for then calculating metrics from
    this data.
    """

    _horizon_n_activations: int
    """Horizon in number of activations."""

    _horizon_steps: int
    """Horizon in number of steps."""

    _steps_since_last_calculated: int
    """Steps since last calculated."""

    _neuron_activity: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    """Neuron activity since inception."""

    _thresholds: list[float]
    """Thresholds for almost dead neurons."""

    _n_components: int
    """Number of components."""

    _n_learned_features: int
    """Number of learned features."""

    @property
    def _dead_count(self) -> Int[Tensor, Axis.COMPONENT]:
        """Dead count."""
        dead_bool_mask: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)] = (
            self._neuron_activity == 0
        )
        return dead_bool_mask.sum(-1)

    @property
    def _dead_fraction(self) -> Float[Tensor, Axis.COMPONENT]:
        """Dead fraction."""
        return self._dead_count / self._n_learned_features

    @property
    def _alive_count(self) -> Int[Tensor, Axis.COMPONENT]:
        """Alive count."""
        alive_bool_mask: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)] = (
            self._neuron_activity > 0
        )

        return alive_bool_mask.sum(-1)

    def _almost_dead(self, threshold: float) -> Int[Tensor, Axis.COMPONENT]:
        """Almost dead count."""
        threshold_in_activations: float = threshold * self._horizon_n_activations

        almost_dead_bool_mask: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)] = (
            self._neuron_activity < threshold_in_activations
        )

        return almost_dead_bool_mask.sum(-1)

    @property
    def _activity_histogram(self) -> list[wandb.Histogram]:
        """Activity histogram."""
        numpy_neuron_activity: Float[
            np.ndarray, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = self._neuron_activity.cpu().numpy()

        np_histograms = [np.histogram(activity) for activity in numpy_neuron_activity]

        return [wandb.Histogram(np_histogram=histogram) for histogram in np_histograms]

    @property
    def _log_activity_histogram(self) -> list[wandb.Histogram]:
        """Log activity histogram."""
        log_epsilon = 0.1  # To avoid log(0)
        log_neuron_activity: Float[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.log(self._neuron_activity + log_epsilon)

        numpy_log_neuron_activity: Float[
            np.ndarray, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = log_neuron_activity.cpu().numpy()

        np_histograms = [np.histogram(activity) for activity in numpy_log_neuron_activity]

        return [wandb.Histogram(np_histogram=histogram) for histogram in np_histograms]

    @property
    def metric_results(self) -> list[MetricResult]:
        """Metric results."""
        metric_location = MetricLocation.TRAIN
        name = "learned_neuron_activity"

        results = [
            MetricResult(
                component_wise_values=self._dead_count,
                location=metric_location,
                name=name,
                postfix=f"dead_over_{self._horizon_n_activations}_activations",
            ),
            MetricResult(
                component_wise_values=self._alive_count,
                location=metric_location,
                name=name,
                postfix=f"alive_over_{self._horizon_n_activations}_activations",
            ),
            MetricResult(
                component_wise_values=self._activity_histogram,
                location=metric_location,
                name=name,
                postfix=f"activity_histogram_over_{self._horizon_n_activations}_activations",
                aggregate_approach=None,  # Don't show aggregate across components
            ),
            MetricResult(
                component_wise_values=self._log_activity_histogram,
                location=metric_location,
                name=name,
                postfix=f"log_activity_histogram_over_{self._horizon_n_activations}_activations",
                aggregate_approach=None,  # Don't show aggregate across components
            ),
        ]

        threshold_results = [
            MetricResult(
                component_wise_values=self._almost_dead(threshold),
                location=metric_location,
                name=name,
                postfix=f"almost_dead_{threshold:.1e}_over_{self._horizon_n_activations}_activations",
            )
            for threshold in self._thresholds
        ]

        return results + threshold_results

    def __init__(
        self,
        approximate_activation_horizon: int,
        n_components: int,
        n_learned_features: int,
        thresholds: list[float],
        train_batch_size: int,
    ) -> None:
        """Initialise the neuron activity horizon data.

        Args:
            approximate_activation_horizon: Approximate activation horizon.
            n_components: Number of components.
            n_learned_features: Number of learned features.
            thresholds: Thresholds for almost dead neurons.
            train_batch_size: Train batch size.
        """
        self._steps_since_last_calculated = 0
        self._neuron_activity = torch.zeros((n_components, n_learned_features), dtype=torch.int64)
        self._thresholds = thresholds
        self._n_components = n_components
        self._n_learned_features = n_learned_features

        # Get a precise activation_horizon
        self._horizon_steps = approximate_activation_horizon // train_batch_size
        self._horizon_n_activations = self._horizon_steps * train_batch_size

    def step(
        self, neuron_activity: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    ) -> list[MetricResult]:
        """Step the neuron activity horizon data.

        Args:
            neuron_activity: Neuron activity.

        Returns:
            Dictionary of metrics (or empty dictionary if no metrics are ready to be logged).
        """
        self._steps_since_last_calculated += 1
        self._neuron_activity += neuron_activity.cpu()

        if self._steps_since_last_calculated >= self._horizon_steps:
            result = [*self.metric_results]
            self._steps_since_last_calculated = 0
            self._neuron_activity = torch.zeros_like(self._neuron_activity)
            return result

        return []


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
        n_learned_features = data.learned_activations.shape[-1]
        n_components = data.learned_activations.shape[-2]

        for horizon in self._approximate_horizons:
            # Don't add horizons that are smaller than the train batch size
            if horizon < train_batch_size:
                continue

            self._data.append(
                NeuronActivityHorizonData(
                    approximate_activation_horizon=horizon,
                    n_components=n_components,
                    n_learned_features=n_learned_features,
                    thresholds=self._thresholds,
                    train_batch_size=train_batch_size,
                )
            )

        self._initialised = True

    def calculate(self, data: TrainMetricData) -> list[MetricResult]:
        """Calculate the neuron activity metrics.

        Args:
            data: Resample metric data.

        Returns:
            Dictionary of metrics.
        """
        if not self._initialised:
            self.initialise_horizons(data)

        fired_count: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)] = (
            data.learned_activations > 0
        ).sum(dim=0)

        horizon_specific_logs: list[list[MetricResult]] = [
            horizon_data.step(fired_count) for horizon_data in self._data
        ]

        # Flatten and return
        return [log for logs in horizon_specific_logs for log in logs]
