"""Test the Neel C4 Tokenized collate function."""
from datasets import load_dataset

from sparse_autoencoder.src_data.datasets.neel_c4_tokenized import (
    collate_neel_c4_tokenized,
)


def test_collate_neel_c4_tokenized():
    """Test the collate result is shaped as expected."""

    dataset = load_dataset(
        "NeelNanda/c4-code-tokenized-2b",
        streaming=True,
        split="train",
        keep_in_memory=True,
    )

    dataset_iter = iter(dataset)
    first_item = next(dataset_iter)

    tokens = collate_neel_c4_tokenized([first_item])

    # The dataset is all 1024 tokens per batch item
    assert tokens.shape[1] == 1024
