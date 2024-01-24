"""L2 Reconstruction loss."""
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torchmetrics import Metric

from sparse_autoencoder.tensor_types import Axis


class L2ReconstructionLoss(Metric):
    """L2 Reconstruction loss (MSE).

    L2 reconstruction loss is calculated as the sum squared error between each each input vector
    and it's corresponding decoded vector. The original paper found that models trained with some
    loss functions such as cross-entropy loss generally prefer to represent features
    polysemantically, whereas models trained with L2 may achieve the same loss for both
    polysemantic and monosemantic representations of true features.

    Example:
        >>> import torch
        >>> loss = L2ReconstructionLoss(num_components=1)
        >>> source_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [4., 2.] # Component 1
        ...     ],
        ...     [ # Batch 2
        ...         [2., 0.] # Component 1
        ...     ]
        ... ])
        >>> decoded_activations = torch.tensor([
        ...     [ # Batch 1
        ...         [2., 0.] # Component 1 (MSE of 4)
        ...     ],
        ...     [ # Batch 2
        ...         [0., 0.] # Component 1 (MSE of 2)
        ...     ]
        ... ])
        >>> loss.forward(
        ...     decoded_activations=decoded_activations, source_activations=source_activations
        ... )
        tensor(3.)
    """

    # Torchmetrics settings
    is_differentiable: bool | None = True
    higher_is_better = False
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
        if keep_batch_dim and not isinstance(self.sum_activation_vectors_mse, list):
            self.add_state(
                "sum_activation_vectors_mse",
                default=[],
                dist_reduce_fx="sum",
            )
        elif not isinstance(self.sum_activation_vectors_mse, Tensor):
            self.add_state(
                "sum_activation_vectors_mse",
                default=torch.zeros(self._num_components),
                dist_reduce_fx="sum",
            )

    # State
    sum_activation_vectors_mse: Float[Tensor, Axis.COMPONENT_OPTIONAL] | list[
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
        """Initialise the L2 reconstruction loss."""
        super().__init__()
        self._num_components = num_components
        self.keep_batch_dim = keep_batch_dim

        self.add_state(
            "num_activation_vectors",
            default=torch.tensor(0, dtype=torch.int64),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        **kwargs,  # type: ignore # noqa: ANN003, ARG002 (allows combining with other metrics)
    ) -> None:
        """Update the metric state.

        If we're keeping the batch dimension, we simply take the mse of the activations
        (over the features dimension) and then append this tensor to a list. Then during compute we
        just concatenate and return this list. This is useful for e.g. getting L1 loss by batch item
        when resampling neurons (see the neuron resampler for details).

        By contrast if we're averaging over the batch dimension, we sum the activations over the
        batch dimension during update (on each process), and then divide by the number of activation
        vectors on compute to get the mean.

        Args:
            decoded_activations: The decoded activations from the autoencoder.
            source_activations: The source activations from the autoencoder.
            **kwargs: Ignored keyword arguments (to allow use with other metrics in a collection).
        """
        mse: Float[Tensor, Axis.COMPONENT_OPTIONAL] = (
            (decoded_activations - source_activations).pow(2).mean(dim=-1)
        )

        if isinstance(self.sum_activation_vectors_mse, list):  # If keeping the batch dimension
            self.sum_activation_vectors_mse.append(mse)

        else:
            self.sum_activation_vectors_mse += mse.sum(dim=0)
            self.num_activation_vectors += source_activations.shape[0]

    def compute(self) -> Float[Tensor, Axis.COMPONENT_OPTIONAL]:
        """Compute the metric."""
        if isinstance(self.sum_activation_vectors_mse, list):  # If keeping the batch dimension
            return torch.cat(self.sum_activation_vectors_mse)

        return self.sum_activation_vectors_mse / self.num_activation_vectors
