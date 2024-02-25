"""Source model tests."""
from einops import rearrange
import torch
from transformer_lens import HookedTransformer

from sparse_autoencoder.source_model.model import GenerateActivationsSourceModel


def test_source_model_activations_match_cache() -> None:
    """Test that the activations match running the underlying model with cache."""
    input_tokens = torch.tensor([[123, 124], [125, 126]])
    model_name = "tiny-stories-instruct-1M"
    cache_name = "blocks.0.hook_mlp_out"

    # Get the cache
    model = HookedTransformer.from_pretrained(model_name)
    _output, cache = model.run_with_cache(input_tokens)
    cached_activations = cache[cache_name]

    # Instead run with a source model
    source_model = GenerateActivationsSourceModel(model_name, ["blocks.0.hook_mlp_out"])
    source_model_activations = source_model.forward(input_tokens)

    reshaped_cached = rearrange(cached_activations, "b p f -> (b p) f").unsqueeze(1)
    assert torch.allclose(source_model_activations, reshaped_cached)
