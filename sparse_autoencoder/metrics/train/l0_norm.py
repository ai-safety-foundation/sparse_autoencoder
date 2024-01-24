"""L0 norm sparsity metric."""
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class L0NormMetric(Metric):
    """Learned activations L0 norm metric.

    The L0 norm is the number of non-zero elements in a learned activation vector, averaged over the
    number of activation vectors.

    Examples:
        >>> metric = L0NormMetric()
        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 0., 1.] # Component 1: learned features (2 active neurons)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 1., 0.] # Component 1: learned features (1 active neuron)
        ...     ]
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
    active_neurons_count: Int64[Tensor, Axis.COMPONENT_OPTIONAL]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(self, num_components: PositiveInt = 1) -> None:
        """Initialize the metric."""
        super().__init__()

        self.add_state(
            "active_neurons_count",
            default=torch.empty(num_components, dtype=torch.int64),
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
            Tensor, Axis.names(Axis.PROCESS_BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        **kwargs,  # type: ignore # noqa: ANN003, ARG002 (allows combining with other metrics)
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
        """Compute the metric."""
        return self.active_neurons_count / self.num_activation_vectors
