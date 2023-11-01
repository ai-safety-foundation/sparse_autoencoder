"""Store Activations Hook Tests."""
from functools import partial

from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.list_store import ListActivationStore
from sparse_autoencoder.src_model.store_activations_hook import store_activations_hook


def test_hook_stores_activations():
    """Test that the hook stores activations correctly."""
    store = ListActivationStore()
    model = HookedTransformer.from_pretrained("tiny-stories-1M")

    model.add_hook(
        "blocks.1.mlp.hook_post", partial(store_activations_hook, store=store)
    )

    tokens = model.to_tokens("Hello world")
    model.forward(tokens, stop_at_layer=2)

    number_of_tokens = tokens.numel()
    mlp_size: int = model.cfg.d_mlp

    assert len(store) == number_of_tokens
    assert store[0].shape[0] == mlp_size
