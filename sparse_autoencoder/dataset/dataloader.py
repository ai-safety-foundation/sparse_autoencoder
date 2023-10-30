"""Create a DataLoader with pre-processed data."""
from typing import Callable

from datasets import IterableDataset, load_dataset
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


def collate_pile(
    batch: list[dict[str, str]], tokenizer: PreTrainedTokenizerBase
) -> Int[Tensor, "batch pos"]:
    """Collate Function for the Pile dataset.

    To be used as a :class:`torch.DataLoader` collate function.

    Examples:

    You can create a collate function for :func:`create_dataloader` as follows:

        >>> from transformers import AutoTokenizer
        >>> from functools import partial
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token="<|endoftext|>")
        >>> collate_fn = partial(collate_pile, tokenizer=tokenizer)

    Args:
        batch: Batch of data from the Pile dataset.
        tokenizer: HuggingFace tokenizer to use.

    Returns:
        Batch of tokenized prompts.
    """
    texts = [item["text"] for item in batch]
    return tokenizer(texts, return_tensors="pt", padding=True, truncation=True)[
        "input_ids"
    ]


def create_dataloader(
    dataset_name: str,
    collate_fn: Callable[[list], Int[Tensor, "batch pos"]],
    dataset_split: str = "train",
    batch_size: int = 1024,
    shuffle_buffer_size: int = 10_000,
    random_seed: int = 0,
    num_workers: int = 2,
) -> DataLoader:
    """Create a DataLoader with tokenized data.

    Creates a DataLoader with a [HuggingFace Dataset](https://huggingface.co/datasets).

    Supports distributed training across GPUs with `torch.nn.DataParallel`, but not across nodes.

    Examples:

    You can create a dataloader with the GPT2 tokenizer and pile uncopyright dataset as follows:

    >>> from transformers import AutoTokenizer
    >>> from functools import partial
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token="<|endoftext|>")
    >>> collate_fn = partial(collate_pile, tokenizer=tokenizer)
    >>> dataloader = create_dataloader(
    ...     "monology/pile-uncopyrighted",
    ...     collate_fn,
    ...     shuffle_buffer_size=2, # In practice this should be 10_000 or more.
    ...     random_seed=0
    ... )
    >>> print(next(iter(dataloader)).shape)
    torch.Size([1024, 1024])

    Args:
        dataset_name: HuggingFace dataset name.
        collate_fn: Function to process a batch of data from the dataset & return a batch of
            tokenized prompts. See :func:`collate_pile` for an example.
        batch_size: Number of prompts to process at once.
        shuffle_buffer_size: Minimum number of prompts to shuffle at once. The DataLoader will
            download this many prompts first and then keep at least this number in memory so
            that there are sufficient numbers of prompts available to shuffle. If the
            HuggingFace dataset is sharded, the DataLoader will also shuffle the shard order.
        random_seed: Random seed used for shuffling prompts.
        num_workers: Number of CPU workers used for loading data. This should be greater than 1
            and less than the number of CPU cores available.


    Returns:
        DataLoader with tokenized data.
    """
    dataset: IterableDataset = load_dataset(
        dataset_name, streaming=True, split=dataset_split
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
