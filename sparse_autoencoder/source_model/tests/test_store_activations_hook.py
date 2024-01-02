"""Store Activations Hook Tests."""
from functools import partial

from jaxtyping import Int
import pytest
import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.source_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.integration_test()
def test_hook_stores_activations() -> None:
    """Test that the hook stores activations correctly."""
    store = TensorActivationStore(max_items=100, n_neurons=256, n_components=1)

    model = HookedTransformer.from_pretrained("tiny-stories-1M")

    model.add_hook(
        "blocks.0.mlp.hook_post",
        partial(store_activations_hook, store=store),
    )

    tokens: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)] = model.to_tokens(
        "Hello world"
    )
    logits = model.forward(tokens, stop_at_layer=2)  # type: ignore

    n_of_tokens = tokens.numel()
    mlp_size: int = model.cfg.d_mlp  # type: ignore

    assert len(store) == n_of_tokens
    assert store[0, 0].shape[0] == mlp_size
    assert torch.is_tensor(logits)  # Check the forward pass completed
