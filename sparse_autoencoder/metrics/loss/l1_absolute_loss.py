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
    sum_learned_activations: Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)] | list[
        Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]
    ]
    num_activation_vectors: Int64[Tensor, Axis.SINGLE_ITEM]

    @validate_call
    def __init__(
        self,
        num_components: PositiveInt = 1,
        *,
        keep_batch_dim: bool = False,
    ) -> None:
        """Initialize the metric.

        Args:
            num_components: Number of components.
            keep_batch_dim: Whether to keep the batch dimension in the loss output.
        """
        super().__init__()

        # Add the state
        self.add_state(
            "sum_learned_activations",
            default=[] if keep_batch_dim else torch.zeros(num_components),  # See `update` method
            dist_reduce_fx="sum",
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

        If we're keeping the batch dimension, we simply take the absolute sum of the activations
        (over the features dimension) and then append this tensor to a list. Then during compute we
        just concatenate and return this list. This is useful for e.g. getting L1 loss by batch item
        when resampling neurons (see the neuron resampler for details).

        By contrast if we're averaging over the batch dimension, we sum the activations over the
        batch dimension during update (on each process), and then divide by the number of activation
        vectors on compute to get the mean.

        Args:
            learned_activations: Learned activations (intermediate activations in the autoencoder).
        """
        absolute_loss = torch.abs(learned_activations).sum(dim=-1)

        if isinstance(self.sum_learned_activations, list):  # If keeping the batch dimension
            self.sum_learned_activations.append(absolute_loss)
        else:
            self.sum_learned_activations += absolute_loss.sum(dim=0)
            self.num_activation_vectors += learned_activations.shape[0]

    def compute(self) -> Tensor:
        """Compute the metric."""
        if isinstance(self.sum_learned_activations, list):  # If keeping the batch dimension
            return torch.cat(self.sum_learned_activations)

        return self.sum_learned_activations / self.num_activation_vectors
