"""Loss function for the Sparse Autoencoder."""
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn.functional import mse_loss


def reconstruction_loss(
    input_activations: Float[Tensor, "*batch input_activations"],
    output_activations: Float[Tensor, "*batch input_activations"],
) -> Tensor:
    """Reconstruction Loss (MSE).

    MSE reconstruction loss is calculated as the mean squared error between each each input vector
    and it's corresponding decoded vector. The original paper found that models trained with some
    loss functions such as cross-entropy loss generally prefer to represent features
    polysemantically, whereas models trained with MSE may achieve the same loss for both
    polysemantic and monosemantic representations of true features.

    Examples:

    >>> input_activations = torch.tensor([[3.0, 4]])
    >>> output_activations = torch.tensor([[1.0, 5]])
    >>> reconstruction_loss(input_activations, output_activations)
    tensor(2.5000)

    Args:
        input_activations: Input activations.
        output_activations: Reconstructed activations.

    Returns:
        Mean Squared Error reconstruction loss.
    """
    return mse_loss(input_activations, output_activations, reduction="mean")


def l1_loss(learned_activations: Float[Tensor, "*batch learned_activations"]) -> Tensor:
    """L1 Loss on Learned Activations

    L1 loss penalty is the absolute sum of the learned activations. The L1 penality is this
    multiplied by the l1_coefficient (designed to encourage sparsity).

    Examples:

    >>> learned_activations = torch.tensor([[2.0, -3]])
    >>> l1_loss(learned_activations)
    tensor([5.])

    Args:
        learned_activations: Activations from the hidden layer.

    Returns:
        L1 loss on learned activations.
    """
    return torch.abs(learned_activations).sum(dim=-1)


def sae_training_loss(
    reconstruction_loss_mse: Tensor,
    l1_loss_learned_activations: Tensor,
    l1_coefficient: float,
) -> Tensor:
    """Loss Function for the Sparse Autoencoder.

    The original paper used L2 reconstruction loss, plus l1 loss on the hidden (learned)
    activations.

    https://transformer-circuits.pub/2023/monosemantic-features/index.html#setup-autoencoder-motivation

    Examples:

    >>> reconstruction_loss_mse = torch.tensor([2.5000])
    >>> l1_loss_learned_activations = torch.tensor([1.])
    >>> l1_coefficient = 0.5
    >>> sae_training_loss(reconstruction_loss_mse, l1_loss_learned_activations, l1_coefficient)
    tensor(3.)

    Args:
        reconstruction_loss_mse: MSE reconstruction loss.
        l1_loss_learned_activations: L1 loss on learned activations.
        l1_coefficient: L1 coefficient. The original paper experimented with L1 coefficients of
            [0.01, 0.008, 0.006, 0.004, 0.001]. They used 250 tokens per prompt, so as an
            approximate guide if you use e.g. 2x this number of tokens you might consider using 0.5x
            the l1 coefficient.

    Returns:
        Overall training loss.
    """
    total_loss = reconstruction_loss_mse + l1_loss_learned_activations * l1_coefficient
    return total_loss.sum()
