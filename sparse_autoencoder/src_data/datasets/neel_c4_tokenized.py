"""Neel C4 Tokenized Dataset.

This dataset was used to train [Neel Nanda's GeLU
models](https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html). These
are known in TransformerLens as `gelu-1l` to `gelu-4l`. The dataset contains code. Each batch item
has exactly the same number of tokens, which makes for easy parsing.
"""

import torch

from sparse_autoencoder.src_data.src_data import (
    CollateResponse,
    CollateResponseMask,
    CollateResponseTokens,
)


def collate_neel_c4_tokenized(
    batch: list[dict[str, list[int]]],
) -> CollateResponse:
    """Collate Function for Neel's C4 Tokenized dataset.

    Args:
        batch: Batch of data from the dataset.

    Returns:
        Batch of tokenized prompts, along with their attention masks (1s for tokens to keep and 0s
            for padding tokens).
    """
    # The batch of data is a list of dicts, each with a "tokens" key containing a list of tokens.
    tokens: list[list[int]] = [i["tokens"] for i in batch]
    tokenized: CollateResponseTokens = torch.tensor(tokens)

    # In this dataset there are no padding tokens, so the mask is just a tensor of 1s.
    mask: CollateResponseMask = torch.ones_like(tokenized)

    return tokenized, mask
