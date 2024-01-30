"""L0 norm sparsity metric."""
from typing import Any

from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


class L0NormMetric(Metric):
    """Learned activations L0 norm metric.

    The L0 norm is the number of non-zero elements in a learned activation vector, averaged over the
    number of activation vectors.

    Examples:
        >>> metric = L0NormMetric()
        >>> learned_activations = torch.tensor([
        ...     [1., 0., 1.], # Batch 1 (single component): learned features (2 active neurons)
        ...     [0., 1., 0.]  # Batch 2 (single component): learned features (1 active neuron)
        ... ])
        >>> metric.forward(learned_activations)
        tensor(1.5000)

        With 2 components, the metric will return the average number of active (non-zero)
        neurons as a 1d tensor.

        >>> metric = L0NormMetric(num_components=2)
        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 0., 1.], # Component 1: learned features (2 active neurons)
        ...         [1., 0., 1.]  # Component 2: learned features (2 active neurons)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 1., 0.], # Component 1: learned features (1 active neuron)
        ...         [1., 0., 1.]  # Component 2: learned features (2 active neurons)
        ...     ]
        ... ])
        >>> metric.forward(learned_activations)
        tensor([1.5000, 2.0000])
    """

    # Torchmetrics settings
    is_differentiable: bool | None = False
    full_state_update: bool | None = False
    plot_lower_bound: float | None = 0.0

    # State
    active_neurons_count: Float[Tensor, Axis.COMPONENT_OPTIONAL]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(self, num_components: PositiveInt | None = None) -> None:
        """Initialize the metric."""
        super().__init__()

        self.add_state(
            "active_neurons_count",
            default=torch.zeros(shape_with_optional_dimensions(num_components), dtype=torch.float),
            dist_reduce_fx="sum",  # Float is needed for dist reduce to work
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
        self.num_activation_vectors += learned_activations.shape[0]

        self.active_neurons_count += torch.count_nonzero(learned_activations, dim=-1).sum(
            dim=0, dtype=torch.int64
        )

    def compute(
        self,
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)]:
        """Compute the metric.

        Note that torchmetrics converts shape `[0]` tensors into scalars (shape `0`).
        """
        return self.active_neurons_count / self.num_activation_vectors
