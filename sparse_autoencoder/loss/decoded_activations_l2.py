"""L2 Reconstruction loss."""
from typing import final

from torch.nn.functional import mse_loss

from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
    TrainBatchStatistic,
)


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
        >>> loss(input_activations, unused_activations, output_activations)
        (tensor(11.), {'l2_reconstruction_loss': 11.0})
    """

    def log_name(self) -> str:
        """Log name.

        Returns:
            Name of the loss module for logging.
        """
        return "l2_reconstruction_loss"

    def forward(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,  # noqa: ARG002
        decoded_activations: InputOutputActivationBatch,
    ) -> TrainBatchStatistic:
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

        # Sum over just the features dimension (i.e. batch itemwise loss). Note this is sum rather
        # than mean to be consistent with L1 loss (and thus make the l1 coefficient stable to number
        # of features).
        return square_error_loss.sum(dim=-1)
