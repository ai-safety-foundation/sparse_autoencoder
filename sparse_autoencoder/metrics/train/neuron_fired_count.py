"""Neuron fired count metric."""
from jaxtyping import Bool, Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


class NeuronFiredCountMetric(Metric):
    """Neuron activity metric.

    Example:
        >>> metric = NeuronFiredCountMetric(num_learned_features=3)
        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 0., 1.] # Component 1: learned features (2 active neurons)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 0., 0.] # Component 1: learned features (0 active neuron)
        ...     ]
        ... ])
        >>> metric.forward(learned_activations)
        tensor([[1, 0, 1]])
    """

    # Torchmetrics settings
    is_differentiable: bool | None = True
    full_state_update: bool | None = True
    plot_lower_bound: float | None = 0.0

    # State
    neuron_fired_count: Int64[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]

    @validate_call
    def __init__(
        self,
        num_learned_features: PositiveInt,
        num_components: PositiveInt | None = None,
    ) -> None:
        """Initialise the metric.

        Args:
            num_learned_features: Number of learned features.
            num_components: Number of components.
        """
        super().__init__()
        self.add_state(
            "neuron_fired_count",
            default=torch.empty(
                shape_with_optional_dimensions(num_components, num_learned_features),
                dtype=torch.int64,
            ),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        learned_activations: Float[
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        **kwargs,  # type: ignore # noqa: ANN003, ARG002 (allows combining with other metrics)
    ) -> None:
        """Update the metric state.

        Args:
            learned_activations: The learned activations.
            **kwargs: Ignored keyword arguments (to allow use with other metrics in a collection).
        """
        neuron_has_fired: Bool[
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ] = torch.gt(learned_activations, 0)

        self.neuron_fired_count += neuron_has_fired.sum(dim=0, dtype=torch.int64)

    def compute(self) -> Int64[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]:
        """Compute the metric."""
        return self.neuron_fired_count
