"""TransformerLens Hook for storing activations."""
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint

from sparse_autoencoder.activation_store.base_store import ActivationStore


def store_activations_hook(
    value: Float[Tensor, "*any neuron"],
    hook: HookPoint,  # pylint: disable=unused-argument
    store: ActivationStore,
):
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

    Next we can add the hook to specific neurons (in this case the first MLP neurons), and do a
    forward pass.

    >>> model.add_hook(
    ...     "blocks.0.mlp.hook_post", partial(store_activations_hook, store=store)
    ... )
    >>> _logits = model.forward("Hello world") # "Hello world" is 3 tokens
    >>> len(store)
    3

    Args:
        value: The activations to store.
        hook: The hook point.
        store: The activation store. This should be pre-initialised with `functools.partial`.
    """
    store.extend(value)
