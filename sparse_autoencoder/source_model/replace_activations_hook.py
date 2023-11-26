"""Replace activations hook."""
from typing import TYPE_CHECKING

from torch import Tensor
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.autoencoder.abstract_autoencoder import AbstractAutoencoder


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import InputOutputActivationBatch


def replace_activations_hook(
    value: Tensor,
    hook: HookPoint,  # noqa: ARG001
    sparse_autoencoder: AbstractAutoencoder,
) -> Tensor:
    """Replace activations hook.

    Args:
        value: The activations to replace.
        hook: The hook point.
        sparse_autoencoder: The sparse autoencoder. This should be pre-initialised with
            `functools.partial`.

    Returns:
        Replaced activations.
    """
    # Squash to just have a "*items" and a "batch" dimension
    original_shape = value.shape
    squashed_value: InputOutputActivationBatch = value.view(-1, value.size(-1))

    # Get the output activations from a forward pass of the SAE
    _learned_activations, output_activations = sparse_autoencoder.forward(squashed_value)

    # Reshape to the original shape
    return output_activations.view(*original_shape)
