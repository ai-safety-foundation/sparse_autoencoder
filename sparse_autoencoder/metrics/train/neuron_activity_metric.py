"""Neuron activity metric.

Logs the number of dead and alive neurons at various horizons. Also logs histograms of neuron
activity, and the number of neurons that are almost dead.
"""
from jaxtyping import Float, Int64
from pydantic import NonNegativeFloat, PositiveInt, validate_call
import torch
from torch import Tensor

from sparse_autoencoder.metrics.train.abstract_train_metric import (
    AbstractTrainMetric,
)
from sparse_autoencoder.tensor_types import Axis


DEFAULT_THRESHOLDS: list[float] = [0.0, 1e-5]


class NeuronActivityMetric(AbstractTrainMetric):
    """Neuron activity metric.

    Example:
        With a single component and a horizon of 2 activations, the metric will return nothing
        after the first activation is added and then computed, and then return the number of dead
        neurons after the second activation is added (with update). The breakdown by component isn't
        shown here as there is just one component.

        >>> learned_activations = torch.tensor([[[0.3, 0.4, 0]]])
        >>> input_activations = torch.ones(1, 1, 2)
        >>> metric = NeuronActivityMetric(["mlp1"], n_learned_features=3, horizon=2)
        >>> metric.update(input_activations, learned_activations, input_activations)
        >>> metric.compute()
        {}

        >>> metric.update(input_activations, learned_activations, input_activations)
        >>> metric.compute()
        {'neuron_activity/dead': tensor(1.), 'neuron_activity/almost_dead_1e-05': tensor(1.)}

    Example:
        The threshold can be changed (e.g. here to 0.5), to determine when neurons are considered
        "almost dead". Neurons are considered almost dead if they fire less than the threshold
        portion of activations (over the horizon).

        >>> input_activations = torch.ones(1, 2, 2)
        >>> metric = NeuronActivityMetric(
        ...     ["mlp1", "mlp2"],
        ...     n_learned_features=3,
        ...     horizon=2,
        ...     thresholds=[0.0, 0.5]
        ...     )
        >>> learned_activations = torch.tensor([[[0.3, 0.3, 0], [0.3, 0, 0]]])
        >>> metric.update(input_activations, learned_activations, input_activations)
        >>> learned_activations = torch.tensor([[[0.3, 0, 0], [0.3, 0, 0]]])
        >>> metric.update(input_activations, learned_activations, input_activations)
        >>> metric.compute()
        {'neuron_activity/dead/mlp1': tensor(1),
        'neuron_activity/dead/mlp2': tensor(2),
        'neuron_activity/dead': tensor(1.5000),
        'neuron_activity/almost_dead_0.5/mlp1': tensor(2),
        'neuron_activity/almost_dead_0.5/mlp2': tensor(2),
        'neuron_activity/almost_dead_0.5': tensor(2.)}

    """

    # Persist state across multiple batches
    full_state_update: bool | None = True

    _horizon: int
    _thresholds: list[float]
    _n_components: int

    neuron_activity: Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]
    num_activations: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(
        self,
        component_names: list[str],
        n_learned_features: PositiveInt,
        horizon: PositiveInt,
        thresholds: list[NonNegativeFloat] = DEFAULT_THRESHOLDS,
    ) -> None:
        """Initialise the neuron activity metric.

        time `calculate` is called.

        Args:
            component_names: Names of the components.
            n_learned_features: Number of learned features.
            horizon: Horizon in number of activations.
            thresholds: Thresholds for dead neurons.
        """
        super().__init__(component_names, "neuron_activity")
        self._horizon = horizon
        self._thresholds = thresholds
        self._n_components = len(component_names)

        self.add_state(
            "num_activations", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum"
        )

        self.add_state(
            "neuron_activity",
            default=torch.zeros((self._n_components, n_learned_features), dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        input_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Update the metric state.

        Args:
            input_activations: The input activations.
            learned_activations: The learned activations.
            decoded_activations: The decoded activations.
        """
        self.num_activations += learned_activations.shape[0]

        process_neuron_activity: Float[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)] = (
            (learned_activations > 0).to(dtype=torch.int64).sum(dim=0)
        )

        self.neuron_activity += process_neuron_activity

    def compute(self) -> dict[str, Tensor]:
        """Compute the metric."""
        # Don't return something unless we're at the horizon
        if self.num_activations < self._horizon:
            return {}

        results = {}

        for threshold in self._thresholds:
            # Get the dead neuron counts for a specific threshold
            threshold_activations = threshold * self.num_activations
            threshold_name = "dead" if threshold == 0 else f"almost_dead_{threshold}"
            dead_neurons: Int64[Tensor, Axis.COMPONENT] = torch.sum(
                self.neuron_activity <= threshold_activations, dim=-1
            )

            threshold_results = {}

            # If there is more than one component, create a separate log for each one
            if self._n_components > 1:
                component_results = {
                    f"{self._metric_name}/{threshold_name}/{component}": value
                    for component, value in zip(self._component_names, dead_neurons)
                }
                threshold_results.update(component_results)

            # Also log the average number of dead neurons (across components)
            threshold_results[f"{self._metric_name}/{threshold_name}"] = dead_neurons.mean(
                dim=0, dtype=torch.float
            )

            results.update(threshold_results)

        # Reset the state so that we start collecting neuron activity from scratch again
        self.reset()

        return results
