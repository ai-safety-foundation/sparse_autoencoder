"""Replace activations hook."""
from typing import TYPE_CHECKING

from torch import Tensor
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.autoencoder.abstract_autoencoder import AbstractAutoencoder


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import Axis
from jaxtyping import Float


def replace_activations_hook(
    value: Tensor,
    hook: HookPoint,  # noqa: ARG001
    sparse_autoencoder: AbstractAutoencoder,
    component_idx: int | None = None,
) -> Tensor:
    """Replace activations hook.

    Args:
        value: The activations to replace.
        hook: The hook point.
        sparse_autoencoder: The sparse autoencoder. This should be pre-initialised with
            `functools.partial`.
        component_idx: The component index to replace the activations with, if just replacing
            activations for a single component. Requires the model to have a component axis.

    Returns:
        Replaced activations.

    Raises:
        RuntimeError: If `component_idx` is specified, but the model does not have a component
    """
    # Squash to just have a "*items" and a "batch" dimension
    original_shape = value.shape

    squashed_value: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)] = value.view(
        -1, value.size(-1)
    )

    if component_idx is not None:
        if sparse_autoencoder.n_components is None:
            error_message = (
                "Cannot replace for a specific component, if the model does not have a "
                "component axis."
            )
            raise RuntimeError(error_message)

        # The approach here is to run a forward pass with dummy values for all components other than
        # the one we want to replace. This is done by expanding the inputs to the SAE for a specific
        # component across all components. We then simply discard the activations for all other
        # components.
        expanded_shape = [
            squashed_value.shape[0],
            sparse_autoencoder.n_components,
            squashed_value.shape[-1],
        ]
        expanded = squashed_value.unsqueeze(1).expand(*expanded_shape)

        _learned_activations, output_activations = sparse_autoencoder.forward(expanded)
        component_output_activations = output_activations[:, component_idx]

        return component_output_activations.view(*original_shape)

    # Get the output activations from a forward pass of the SAE
    _learned_activations, output_activations = sparse_autoencoder.forward(squashed_value)

    # Reshape to the original shape
    return output_activations.view(*original_shape)
