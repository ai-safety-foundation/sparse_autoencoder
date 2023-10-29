"""Loss function for the Sparse Autoencoder."""
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss


def loss(
    input_activations: Float[Tensor, "*batch input_activations"],
    learned_activations: Float[Tensor, "*batch learned_activations"],
    output_activations: Float[Tensor, "*batch input_activations"],
    l1_coefficient: Float[Tensor, "*batch"],
) -> Tensor:
    """Loss Function for the Sparse Autoencoder.

    The objective of an autoencoder is to minimize the difference (reconstruction error) between the
    original input and its reconstruction. The original paper used L2 reconstruction loss, plus l1
    loss on the hidden (learned) activations.

    MSE reconstruction loss is calculated as the mean squared error between each each input vector
    and it's corresponding decoded vector. The original paper found that models trained with some
    loss functions such as cross-entropy loss generally prefer to represent features
    polysemantically, whereas models trained with MSE may achieve the same loss for both
    polysemantic and monosemantic representations of true features.

    L1 loss penalty is the absolute sum of the learned activations. This penalty encourages
    sparsity.

    https://transformer-circuits.pub/2023/monosemantic-features/index.html#setup-autoencoder-motivation

    Args:
        input_activations: Input activations. output_activations: Output activations.
        l1_coefficient: L1 coefficient. The original paper experimented with L1 coefficients of
            [0.01, 0.008, 0.006, 0.004, 0.001]. They used 250 tokens per prompt, so as an
            approximate guide if you use e.g. 2x this number of tokens you might consider using 0.5x
            the l1 coefficient.
    """
    mse_reconstruction_loss: Float[Tensor, "*batch"] = mse_loss(
        input_activations, output_activations, reduction="mean"
    )

    l1_loss: Float[Tensor, "*batch"] = torch.abs(learned_activations).sum(dim=-1)
    l1_penalty: Float[Tensor, "*batch"] = l1_coefficient * l1_loss

    return mse_reconstruction_loss + l1_penalty
