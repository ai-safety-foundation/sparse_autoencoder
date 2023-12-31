"""Replace activations hook tests."""
from functools import partial

from jaxtyping import Int
import pytest
import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.source_model.replace_activations_hook import replace_activations_hook
from sparse_autoencoder.tensor_types import Axis


@pytest.mark.integration_test()
def test_hook_replaces_activations() -> None:
    """Test that the hook replaces activations."""
    torch.random.manual_seed(0)
    source_model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    autoencoder = SparseAutoencoder(source_model.cfg.d_model, source_model.cfg.d_model * 2)

    tokens: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)] = source_model.to_tokens(
        "Hello world"
    )
    loss_without_hook = source_model.forward(tokens, return_type="loss")
    loss_with_hook = source_model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[
            (
                "blocks.0.hook_mlp_out",
                partial(replace_activations_hook, sparse_autoencoder=autoencoder),
            )
        ],
    )

    # Check it decrease performance (as the sae is untrained so it will output nonsense).
    assert torch.all(torch.gt(loss_with_hook, loss_without_hook))


@pytest.mark.integration_test()
def test_hook_replaces_activations_2_components() -> None:
    """Test that the hook replaces activations."""
    torch.random.manual_seed(0)
    source_model = HookedTransformer.from_pretrained("tiny-stories-1M", device="cpu")
    autoencoder = SparseAutoencoder(
        source_model.cfg.d_model, source_model.cfg.d_model * 2, n_components=2
    )

    tokens: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)] = source_model.to_tokens(
        "Hello world"
    )
    loss_without_hook = source_model.forward(tokens, return_type="loss")
    loss_with_hook = source_model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[
            (
                "blocks.0.hook_mlp_out",
                partial(replace_activations_hook, sparse_autoencoder=autoencoder, component_idx=1),
            )
        ],
    )

    # Check it decrease performance (as the sae is untrained so it will output nonsense).
    assert torch.all(torch.gt(loss_with_hook, loss_without_hook))
