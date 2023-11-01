"""Source Data.

Gets large amounts of text that can be used as prompts for the source model, to be used in getting
activations.

Note that for shared types, we include the shape in the docstring, as code hints aren't supported 
by jaxtyping.
"""
from typing import Callable

from datasets import IterableDataset, load_dataset
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader

CollateResponseTokens = Int[Tensor, "batch pos"]
"""Collate Response Tokens Type.

Shape [batch, pos].
"""


def create_src_dataloader(
    dataset_name: str,
    collate_fn: Callable[[list], CollateResponseTokens],
    dataset_split: str = "train",
    batch_size: int = 512,
    shuffle_buffer_size: int = 10_000,
    random_seed: int = 0,
    num_workers: int = 2,
) -> DataLoader:
    """Create a DataLoader with tokenized data.

    Creates a DataLoader with a [HuggingFace Dataset](https://huggingface.co/datasets).

    Supports distributed training across GPUs with `torch.nn.DataParallel`, but not across nodes.

    Examples:

    You can create a dataloader with the GPT2 tokenizer and pile uncopyrighted dataset as follows:

    >>> from sparse_autoencoder.src_data.datasets.neel_c4_tokenized import collate_neel_c4_tokenized
    >>> dataloader = create_src_dataloader(
    ...     "NeelNanda/c4-code-tokenized-2b",
    ...     collate_fn=collate_neel_c4_tokenized,
    ...     shuffle_buffer_size=512, # In practice this should be 10_000 or more.
    ...     random_seed=0
    ... )
    >>> print(next(iter(dataloader)).shape)
    torch.Size([512, 1024])

    Args:
        dataset_name: HuggingFace dataset name.
        collate_fn: Function to process a batch of data from the dataset & return a batch of
            tokenized prompts. See :func:`collate_pile` for an example.
        dataset_split: HuggingFace dataset split to use (e.g. `train`).
        batch_size: Number of prompts to process at once.
        shuffle_buffer_size: Minimum number of prompts to shuffle at once. The DataLoader will
            download this many prompts first and then keep at least this number in memory so that
            there are sufficient numbers of prompts available to shuffle. If the HuggingFace dataset
            is sharded, the DataLoader will also shuffle the shard order.
        random_seed: Random seed used for shuffling prompts.
        num_workers: Number of CPU workers used for loading data. This should be greater than 1 and
            less than the number of CPU cores available.

    Returns:
        DataLoader with tokenized data
    """
    dataset: IterableDataset = load_dataset(
        dataset_name,
        streaming=True,
        split=dataset_split,
    )

    # This dataset fills a buffer with buffer_size elements, then randomly samples elements from
    # this buffer, replacing the selected elements with new elements.
    shuffled_dataset = dataset.shuffle(
        seed=random_seed, buffer_size=shuffle_buffer_size
    )

    return DataLoader(
        shuffled_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
