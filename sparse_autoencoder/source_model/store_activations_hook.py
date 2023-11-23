"""TransformerLens Hook for storing activations."""
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.tensor_types import SourceModelActivations


def store_activations_hook(
    value: SourceModelActivations,
    hook: HookPoint,  # noqa: ARG001
    store: ActivationStore,
) -> SourceModelActivations:
    """Store Activations Hook.

    Useful for getting just the specific activations wanted, rather than the full cache.

    Example:
    First we'll need a source model from TransformerLens and an activation store.

    >>> from functools import partial
    >>> from transformer_lens import HookedTransformer
    >>> from sparse_autoencoder.activation_store.list_store import ListActivationStore
    >>> store = ListActivationStore()
    >>> model = HookedTransformer.from_pretrained("tiny-stories-1M")
    Loaded pretrained model tiny-stories-1M into HookedTransformer

    Next we can add the hook to specific neurons (in this case the first MLP neurons), and create
    the tokens for a forward pass.

    >>> model.add_hook(
    ...     "blocks.0.mlp.hook_post", partial(store_activations_hook, store=store)
    ... )
    >>> tokens = model.to_tokens("Hello world")
    >>> tokens.shape
    torch.Size([1, 3])

    Then when we run the model, we should get one activation vector for each token (as we just have
    one batch item). Note we also set `stop_at_layer=1` as we don't need the logits or any other
    activations after the hook point that we've specified (in this case the first MLP layer).

    >>> _output = model.forward("Hello world", stop_at_layer=1) # Change this layer as required
    >>> len(store)
    3

    Args:
        value: The activations to store.
        hook: The hook point.
        store: The activation store. This should be pre-initialised with `functools.partial`.

    Returns:
        Unmodified activations.
    """
    store.extend(value)

    # Return the unmodified value
    return value
