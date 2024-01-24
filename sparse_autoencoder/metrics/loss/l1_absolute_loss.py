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

    # Settings
    _num_components: int

    @property
    def num_components(self) -> int:
        """Number of components."""
        return self._num_components

    _keep_batch_dim: bool

    @property
    def keep_batch_dim(self) -> bool:
        """Whether to keep the batch dimension in the loss output."""
        return self._keep_batch_dim

    @keep_batch_dim.setter
    def keep_batch_dim(self, keep_batch_dim: bool) -> None:
        """Set whether to keep the batch dimension in the loss output.

        When setting this we need to change the state to either a list if keeping the batch
        dimension (so we can accumulate all the losses and concatenate them at the end along this
        dimension). Alternatively it should be a tensor if not keeping the batch dimension (so we
        can sum the losses over the batch dimension during update and then take the mean).

        By doing this in a setter we allow changing of this setting after the metric is initialised.
        """
        self._keep_batch_dim = keep_batch_dim
        self.reset()  # Reset the metric to update the state
        if keep_batch_dim and not isinstance(self.sum_learned_activations, list):
            self.add_state(
                "sum_learned_activations",
                default=[],
                dist_reduce_fx="sum",
            )
        elif not isinstance(self.sum_learned_activations, Tensor):
            self.add_state(
                "sum_learned_activations",
                default=torch.zeros(self._num_components),
                dist_reduce_fx="sum",
            )

    # State
    sum_learned_activations: Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL)] | list[
        Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]
    ] | None = None
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
        self._num_components = num_components
        self.keep_batch_dim = keep_batch_dim

        # Add the state
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
        **kwargs,  # type: ignore # noqa: ANN003, ARG002 (allows combining with other metrics)
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
            **kwargs: Ignored keyword arguments (to allow use with other metrics in a collection).
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
