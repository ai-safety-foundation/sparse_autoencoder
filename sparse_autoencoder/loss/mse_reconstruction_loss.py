"""MSE Reconstruction loss."""
from typing import final

from torch.nn.functional import mse_loss

from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
    TrainBatchStatistic,
)


@final
class MSEReconstructionLoss(AbstractLoss):
    """MSE Reconstruction loss.

    MSE reconstruction loss is calculated as the mean squared error between each each input vector
    and it's corresponding decoded vector. The original paper found that models trained with some
    loss functions such as cross-entropy loss generally prefer to represent features
    polysemantically, whereas models trained with MSE may achieve the same loss for both
    polysemantic and monosemantic representations of true features.

    Example:
        >>> import torch
        >>> loss = MSEReconstructionLoss()
        >>> input_activations = torch.tensor([[5.0, 4], [3.0, 4]])
        >>> output_activations = torch.tensor([[1.0, 5], [1.0, 5]])
        >>> unused_activations = torch.zeros_like(input_activations)
        >>> # Outputs both loss and metrics to log
        >>> loss(input_activations, unused_activations, output_activations)
        (tensor(5.5000), {'MSEReconstructionLoss': 5.5})
    """

    def forward(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,  # noqa: ARG002
        decoded_activations: InputOutputActivationBatch,
    ) -> TrainBatchStatistic:
        """MSE Reconstruction loss (mean across features dimension).

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Loss per batch item.
        """
        square_error_loss = mse_loss(source_activations, decoded_activations, reduction="none")

        # Mean over just the features dimension (i.e. batch itemwise loss)
        return square_error_loss.mean(dim=-1)
