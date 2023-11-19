"""Random Int Dummy Source Data.

For use with tests and simple examples.
"""
import random
from typing import TypedDict, final

from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast

from sparse_autoencoder.source_data.abstract_dataset import (
    SourceDataset,
    TokenizedPrompts,
    TorchTokenizedPrompts,
)


class RandomIntSourceData(TypedDict):
    """Random Int Dummy Source Data."""

    input_ids: list[list[int]]


class RandomIntHuggingFaceDataset(Dataset):
    """Dummy Hugging Face Dataset."""

    def __init__(self, vocab_size: int, context_size: int):
        """Initialize the Random Int Dummy Hugging Face Dataset.

        Args:
            vocab_size: The size of the vocabulary to use.
            context_size: The number of tokens in the context window
        """
        self.vocab_size = vocab_size
        self.context_size = context_size

    def __iter__(self) -> "RandomIntHuggingFaceDataset":  # type: ignore
        """Iter Dunder Method."""
        return self

    def __next__(self) -> dict[str, list[int]]:
        """Next Dunder Method."""
        data = [random.randint(0, self.vocab_size) for _ in range(self.context_size)]  # noqa: S311
        return {"input_ids": data}

    def __len__(self) -> int:
        """Len Dunder Method."""
        return 1000

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        """Get Item."""
        return self.__next__()


@final
class RandomIntDummyDataset(SourceDataset[RandomIntSourceData]):
    """Random Int Dummy Dataset.

    For use with tests and simple examples.
    """

    tokenizer: PreTrainedTokenizerFast

    def preprocess(
        self,
        source_batch: RandomIntSourceData,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts.

        Not implemented for this dummy dataset.
        """
        raise NotImplementedError

    def __init__(
        self,
        context_size: int = 250,
        buffer_size: int = 1000,  # noqa: ARG002
        preprocess_batch_size: int = 1000,  # noqa: ARG002
        dataset_path: str = "dummy",  # noqa: ARG002
        dataset_split: str = "train",  # noqa: ARG002
    ):
        """Initialize the Random Int Dummy dataset.

        Example:
            >>> data = RandomIntDummyDataset()
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
        self.dataset = RandomIntHuggingFaceDataset(50000, context_size=context_size)  # type: ignore

    def get_dataloader(self, batch_size: int) -> DataLoader[TorchTokenizedPrompts]:  # type: ignore
        """Get Dataloader."""
        return DataLoader[TorchTokenizedPrompts](self.dataset, batch_size=batch_size)  # type: ignore
