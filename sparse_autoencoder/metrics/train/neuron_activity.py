"""Neuron activity metric."""
from typing import Annotated, Any

from jaxtyping import Bool, Float, Int64
from pydantic import Field, NonNegativeFloat, PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


class NeuronActivityMetric(Metric):
    """Neuron activity metric.

    Example:
        With a single component and a horizon of 2 activations, the metric will return nothing
        after the first activation is added and then computed, and then return the number of dead
        neurons after the second activation is added (with update). The breakdown by component isn't
        shown here as there is just one component.

        >>> metric = NeuronActivityMetric(num_learned_features=3)
        >>> learned_activations = torch.tensor([
        ...     [1., 0., 1.], # Batch 1 (single component): learned features (2 active neurons)
        ...     [0., 0., 0.]  # Batch 2 (single component): learned features (0 active neuron)
        ... ])
        >>> metric.forward(learned_activations)
        tensor(1)
    """

    # Torchmetrics settings
    is_differentiable: bool | None = False
    full_state_update: bool | None = True
    plot_lower_bound: float | None = 0.0

    # Metric settings
    _threshold_is_dead_portion_fires: NonNegativeFloat

    # State
    neuron_fired_count: Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(
        self,
        num_learned_features: PositiveInt,
        num_components: PositiveInt | None = None,
        threshold_is_dead_portion_fires: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.0,
    ) -> None:
        """Initialise the metric.

        Args:
            num_learned_features: Number of learned features.
            num_components: Number of components.
            threshold_is_dead_portion_fires: Thresholds for counting a neuron as dead (portion of
                activation vectors that it fires for must be less than or equal to this number).
                Commonly used values are 0.0, 1e-5 and 1e-6.
        """
        super().__init__()
        self._threshold_is_dead_portion_fires = threshold_is_dead_portion_fires

        self.add_state(
            "neuron_fired_count",
            default=torch.zeros(
                shape_with_optional_dimensions(num_components, num_learned_features),
                dtype=torch.float,  # Float is needed for dist reduce to work
            ),
            dist_reduce_fx="sum",
        )

        self.add_state(
            "num_activation_vectors",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        **kwargs: Any,  # type: ignore # noqa: ARG002, ANN401 (allows combining with other metrics)
    ) -> None:
        """Update the metric state.

        Args:
            learned_activations: The learned activations.
            **kwargs: Ignored keyword arguments (to allow use with other metrics in a collection).
        """
        # Increment the counter of activations seen since the last compute step
        self.num_activation_vectors += learned_activations.shape[0]

        # Count the number of active neurons in the batch
        neuron_has_fired: Bool[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ] = torch.gt(learned_activations, 0)

        self.neuron_fired_count += neuron_has_fired.sum(dim=0, dtype=torch.float)

    def compute(self) -> Int64[Tensor, Axis.COMPONENT_OPTIONAL]:
        """Compute the metric.

        Note that torchmetrics converts shape `[0]` tensors into scalars (shape `0`).
        """
        threshold_activations: Float[Tensor, Axis.SINGLE_ITEM] = (
            self._threshold_is_dead_portion_fires * self.num_activation_vectors
        )

        return torch.sum(
            self.neuron_fired_count <= threshold_activations, dim=-1, dtype=torch.int64
        )
