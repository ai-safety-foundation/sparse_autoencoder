"""Test the perplexity metric."""
import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.metrics.validate.abstract_validate_metric import ValidationMetricContext
from sparse_autoencoder.metrics.validate.perplexity_metric import PerplexityMetric
from sparse_autoencoder.source_data.text_dataset import GenericTextDataset
from sparse_autoencoder.train.utils import get_model_device


def test_perplexity_metric_with_identity() -> None:
    """Test the perplexity metric."""
    src_model_name = "tiny-stories-1M"
    src_model = HookedTransformer.from_pretrained(src_model_name, dtype="float32")
    src_mlp_width: int = src_model.cfg.d_mlp  # type: ignore

    device = get_model_device(src_model)

    tokenizer: PreTrainedTokenizerBase = src_model.tokenizer  # type: ignore
    source_data = GenericTextDataset(tokenizer=tokenizer, dataset_path="roneneldan/TinyStories")

    autoencoder = SparseAutoencoder(
        n_input_features=src_mlp_width,
        n_learned_features=src_mlp_width,
        geometric_median_dataset=torch.zeros(src_mlp_width),
    ).to(device=device)
    autoencoder.encoder.weight.data = torch.eye(src_mlp_width).to(device=device)
    autoencoder.decoder.weight.data = torch.eye(src_mlp_width).to(device=device)
    autoencoder.encoder.bias.data = torch.zeros(src_mlp_width).to(device=device)

    context = ValidationMetricContext(
        autoencoder=autoencoder,
        source_model=src_model,
        dataset=source_data,
        hook_point="blocks.1.mlp.hook_post",
    )

    perplexity_metric = PerplexityMetric()
    results = perplexity_metric.calculate(context)  # type: ignore
    assert np.isclose(results["SAE loss"], results["Standard loss"], atol=1e-4)
    assert np.isclose(results["KL divergence"], 0.0, atol=1e-4)
