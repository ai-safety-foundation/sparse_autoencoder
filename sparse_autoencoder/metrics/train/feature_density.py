"""Train batch feature density."""
from typing import Any

from jaxtyping import Bool, Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


class FeatureDensityMetric(Metric):
    """Feature density metric.

    Percentage of samples in which each feature was active (i.e. the neuron has "fired"), in a
    training batch.

    Generally we want a small number of features to be active in each batch, so average feature
    density should be low. By contrast if the average feature density is high, it means that the
    features are not sparse enough.

    Example:
        >>> metric = FeatureDensityMetric(num_learned_features=3, num_components=1)
        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 0., 1.] # Component 1: learned features (2 active neurons)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 0., 0.] # Component 1: learned features (0 active neuron)
        ...     ]
        ... ])
        >>> metric.forward(learned_activations)
        tensor([[0.5000, 0.0000, 0.5000]])
    """

    # Torchmetrics settings
    is_differentiable: bool | None = False
    full_state_update: bool | None = True
    plot_lower_bound: float | None = 0.0
    plot_upper_bound: float | None = 1.0

    # State
    neuron_fired_count: Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(
        self, num_learned_features: PositiveInt, num_components: PositiveInt | None = None
    ) -> None:
        """Initialise the metric."""
        super().__init__()

        self.add_state(
            "neuron_fired_count",
            default=torch.zeros(
                size=shape_with_optional_dimensions(num_components, num_learned_features),
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

        self.neuron_fired_count += neuron_has_fired.sum(dim=0, dtype=torch.int64)

    def compute(
        self,
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]:
        """Compute the metric."""
        return self.neuron_fired_count / self.num_activation_vectors
