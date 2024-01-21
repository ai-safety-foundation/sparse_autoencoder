"""L1 (absolute error) loss."""
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class L1AbsoluteLoss(Metric):
    """L1 (absolute error) loss.

    L1 loss penalty is the absolute sum of the learned activations, averaged over the number of
    activation vectors.

    Example:
        >>> l1_loss = L1AbsoluteLoss()
        >>> learned_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [1., 0., 1.] # Component 1: learned features (L1 of 2)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 1., 0.] # Component 1: learned features (L1 of 1)
        ...     ]
        ... ])
        >>> l1_loss.forward(learned_activations=learned_activations)
        tensor(1.5000)
    """

    # Torchmetrics settings
    is_differentiable: bool | None = True
    full_state_update: bool | None = False
    plot_lower_bound: float | None = 0.0

    # State
    sum_learned_activations: Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(self, num_components: PositiveInt = 1) -> None:
        """Initialize the metric."""
        super().__init__()

        self.add_state(
            "sum_learned_activations", default=torch.zeros(num_components), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_activation_vectors",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        *,  # Keyword args so that torchmetrics collections pass just the required args
        learned_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
    ) -> None:
        """Update the metric state.

        Args:
            learned_activations: Learned activations (intermediate activations in the autoencoder).
        """
        self.sum_learned_activations += torch.abs(learned_activations).sum(dim=-1).sum(dim=0)
        self.num_activation_vectors += learned_activations.shape[0]

    def compute(self) -> Float[Tensor, Axis.COMPONENT_OPTIONAL]:
        """Compute the metric."""
        return self.sum_learned_activations / self.num_activation_vectors
