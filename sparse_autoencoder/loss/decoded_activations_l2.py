"""L2 Reconstruction loss."""
from typing import final

from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss

from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.tensor_types import Axis


@final
class L2ReconstructionLoss(AbstractLoss):
    """L2 Reconstruction loss.

    L2 reconstruction loss is calculated as the sum squared error between each each input vector
    and it's corresponding decoded vector. The original paper found that models trained with some
    loss functions such as cross-entropy loss generally prefer to represent features
    polysemantically, whereas models trained with L2 may achieve the same loss for both
    polysemantic and monosemantic representations of true features.

    Example:
        >>> import torch
        >>> loss = L2ReconstructionLoss()
        >>> input_activations = torch.tensor([[5.0, 4], [3.0, 4]])
        >>> output_activations = torch.tensor([[1.0, 5], [1.0, 5]])
        >>> unused_activations = torch.zeros_like(input_activations)
        >>> # Outputs both loss and metrics to log
        >>> loss.forward(input_activations, unused_activations, output_activations)
        tensor([8.5000, 2.5000])
    """

    _reduction: LossReductionType
    """MSE reduction type."""

    def __init__(self, reduction: LossReductionType = LossReductionType.MEAN) -> None:
        """Initialise the L2 reconstruction loss.

        Args:
            reduction: MSE reduction type.
        """
        super().__init__()
        self._reduction = reduction

    def log_name(self) -> str:
        """Log name.

        Returns:
            Name of the loss module for logging.
        """
        return "l2_reconstruction_loss"

    def forward(
        self,
        source_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
        learned_activations: Float[  # noqa: ARG002
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)
        ],
        decoded_activations: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> (
        Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL)]
        | Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]
    ):
        """Calculate the L2 reconstruction loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Loss per batch item.
        """
        square_error_loss = mse_loss(source_activations, decoded_activations, reduction="none")

        match self._reduction:
            case LossReductionType.MEAN:
                return square_error_loss.mean(dim=-1)
            case LossReductionType.SUM:
                return square_error_loss.sum(dim=-1)
            case LossReductionType.NONE:
                return square_error_loss
