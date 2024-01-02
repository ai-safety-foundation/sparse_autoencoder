"""Mock dataset.

For use with tests and simple examples.
"""
from collections.abc import Iterator
from typing import Literal, final

from datasets import IterableDataset
from jaxtyping import Int
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from sparse_autoencoder.source_data.abstract_dataset import (
    SourceDataset,
    TokenizedPrompts,
    TorchTokenizedPrompts,
)


class ConsecutiveIntHuggingFaceDataset(IterableDataset):
    """Consecutive integers Hugging Face dataset for testing.

    Creates a dataset where the first item is [0,1,2...], and the second item is [1,2,3...] and so
    on.
    """

    _data: Int[Tensor, "items context_size"]
    """Generated data."""

    _length: int
    """Size of the dataset."""

    _format: Literal["torch", "list"] = "list"
    """Format of the data."""

    def create_data(self, n_items: int, context_size: int) -> Int[Tensor, "items context_size"]:
        """Create the data.

        Args:
            n_items: The number of items in the dataset.
            context_size: The number of tokens in the context window.

        Returns:
            The generated data.
        """
        rows = torch.arange(n_items).unsqueeze(1)
        columns = torch.arange(context_size).unsqueeze(0)
        return rows + columns

    def __init__(self, context_size: int, vocab_size: int = 50_000, n_items: int = 10_000) -> None:
        """Initialize the mock HF dataset.

        Args:
            context_size: The number of tokens in the context window
            vocab_size: The size of the vocabulary to use.
            n_items: The number of items in the dataset.

        Raises:
            ValueError: If more items are requested than we can create with the vocab size (given
                that each item is a consecutive list of integers and unique).
        """
        self._length = n_items

        # Check we can create the data
        if n_items + context_size > vocab_size:
            error_message = (
                f"n_items ({n_items}) + context_size ({context_size}) must be less than "
                f"vocab_size ({vocab_size})"
            )
            raise ValueError(error_message)

        # Initialise the data
        self._data = self.create_data(n_items, context_size)

    def __iter__(self) -> Iterator:  # type: ignore (HF typing is incorrect)
        """Initialize the iterator.

        Returns:
            Iterator.
        """
        self._index = 0
        return self

    def __next__(self) -> TokenizedPrompts | TorchTokenizedPrompts:
        """Return the next item in the dataset.

        Returns:
            TokenizedPrompts: The next item in the dataset.

        Raises:
            StopIteration: If the end of the dataset is reached.
        """
        if self._index < self._length:
            item = self[self._index]
            self._index += 1
            return item

        raise StopIteration

    def __len__(self) -> int:
        """Len Dunder Method."""
        return self._length

    def __getitem__(self, index: int) -> TokenizedPrompts | TorchTokenizedPrompts:
        """Get Item."""
        item = self._data[index]

        if self._format == "torch":
            return {"input_ids": item}

        return {"input_ids": item.tolist()}

    def with_format(  # type: ignore (only support 2 types)
        self,
        type: Literal["torch", "list"],  # noqa: A002
    ) -> "ConsecutiveIntHuggingFaceDataset":
        """With Format."""
        self._format = type
        return self


@final
class MockDataset(SourceDataset[TokenizedPrompts]):
    """Mock dataset for testing.

    For use with tests and simple examples.
    """

    tokenizer: PreTrainedTokenizerFast

    def preprocess(
        self,
        source_batch: TokenizedPrompts,
        *,
        context_size: int,  # noqa: ARG002
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts."""
        # Nothing to do here
        return source_batch

    @validate_call
    def __init__(
        self,
        context_size: PositiveInt = 250,
        buffer_size: PositiveInt = 1000,  # noqa: ARG002
        preprocess_batch_size: PositiveInt = 1000,  # noqa: ARG002
        dataset_path: str = "dummy",  # noqa: ARG002
        dataset_split: str = "train",  # noqa: ARG002
    ):
        """Initialize the Random Int Dummy dataset.

        Example:
            >>> data = MockDataset()
            >>> first_item = next(iter(data))
            >>> len(first_item["input_ids"])
            250

        Args:
            context_size: The context size to use when returning a list of tokenized prompts.
                *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
                a context size of 250.
            buffer_size: The buffer size to use when shuffling the dataset. As the dataset is
                streamed, this just pre-downloads at least `buffer_size` items and then shuffles
                just that buffer. Note that the generated activations should also be shuffled before
                training the sparse autoencoder, so a large buffer may not be strictly necessary
                here. Note also that this is the number of items in the dataset (e.g. number of
                prompts) and is typically significantly less than the number of tokenized prompts
                once the preprocessing function has been applied.
            preprocess_batch_size: The batch size to use just for preprocessing the dataset (e.g.
                tokenizing prompts).
            dataset_path: The path to the dataset on Hugging Face.
            dataset_split: Dataset split (e.g. `train`).
        """
        self.dataset = ConsecutiveIntHuggingFaceDataset(context_size=context_size)  # type: ignore
        self.context_size = context_size
