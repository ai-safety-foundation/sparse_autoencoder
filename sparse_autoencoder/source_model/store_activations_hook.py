"""TransformerLens Hook for storing activations."""
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.source_model.reshape_activations import (
    ReshapeActivationsFunction,
    reshape_to_last_dimension,
)
from sparse_autoencoder.tensor_types import Axis


def store_activations_hook(
    value: Float[Tensor, Axis.names(Axis.ANY)],
    hook: HookPoint,  # noqa: ARG001
    store: ActivationStore,
    reshape_method: ReshapeActivationsFunction = reshape_to_last_dimension,
    component_idx: int = 0,
) -> Float[Tensor, Axis.names(Axis.ANY)]:
    """Store Activations Hook.

    Useful for getting just the specific activations wanted, rather than the full cache.

    Example:
        First we'll need a source model from TransformerLens and an activation store.

        >>> from functools import partial
        >>> from transformer_lens import HookedTransformer
        >>> from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
        >>> store = TensorActivationStore(max_items=1000, n_neurons=64, n_components=1)
        >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
        Loaded pretrained model tiny-stories-1M into HookedTransformer

        Next we can add the hook to specific neurons (in this case the first MLP neurons), and
        create the tokens for a forward pass.

        >>> model.add_hook(
        ...     "blocks.0.hook_mlp_out", partial(store_activations_hook, store=store)
        ... )
        >>> tokens = model.to_tokens("Hello world")
        >>> tokens.shape
        torch.Size([1, 3])

        Then when we run the model, we should get one activation vector for each token (as we just
        have one batch item). Note we also set `stop_at_layer=1` as we don't need the logits or any
        other activations after the hook point that we've specified (in this case the first MLP
        layer).

        >>> _output = model.forward("Hello world", stop_at_layer=1) # Change this layer as required
        >>> len(store)
        3

    Args:
        value: The activations to store.
        hook: The hook point.
        store: The activation store. This should be pre-initialised with `functools.partial`.
        reshape_method: The method to reshape the activations before storing them.
        component_idx: The component index of the activations to store.

    Returns:
        Unmodified activations.
    """
    reshaped: Float[
        Tensor, Axis.names(Axis.STORE_BATCH, Axis.INPUT_OUTPUT_FEATURE)
    ] = reshape_method(value)

    store.extend(reshaped, component_idx=component_idx)

    # Return the unmodified value
    return value
