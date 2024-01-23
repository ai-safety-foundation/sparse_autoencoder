"""Replace activations hook."""
from typing import TYPE_CHECKING

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DataParallel
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.autoencoder.lightning import LitSparseAutoencoder
from sparse_autoencoder.autoencoder.model import SparseAutoencoder


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import Axis


def replace_activations_hook(
    value: Tensor,
    hook: HookPoint,  # noqa: ARG001
    sparse_autoencoder: SparseAutoencoder
    | DataParallel[SparseAutoencoder]
    | LitSparseAutoencoder
    | Module,
    component_idx: int | None = None,
    n_components: int | None = None,
) -> Tensor:
    """Replace activations hook.

    This should be pre-initialised with `functools.partial`.

    Args:
        value: The activations to replace.
        hook: The hook point.
        sparse_autoencoder: The sparse autoencoder.
        component_idx: The component index to replace the activations with, if just replacing
            activations for a single component. Requires the model to have a component axis.
        n_components: The number of components that the SAE is trained on.

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
        if n_components is None:
            error_message = "The number of model components must be set if component_idx is set."
            raise RuntimeError(error_message)

        # The approach here is to run a forward pass with dummy values for all components other than
        # the one we want to replace. This is done by expanding the inputs to the SAE for a specific
        # component across all components. We then simply discard the activations for all other
        # components.
        expanded_shape = [
            squashed_value.shape[0],
            n_components,
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
