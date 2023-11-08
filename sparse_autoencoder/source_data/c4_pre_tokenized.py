"""Neel Nanda C4 Pre-Tokenized 2B Dataset.

This dataset was used to train [Neel Nanda's GeLU
models](https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html). These
are known in TransformerLens as `gelu-1l` to `gelu-4l`. The dataset is pre-tokenized.
"""
from typing import TypedDict, final

from sparse_autoencoder.source_data.abstract_dataset import (
    SourceDataset,
    TokenizedPrompts,
)


class NeelC4SourceDataBatch(TypedDict):
    """Neel Nanda C4 Pre-Tokenized 2B Dataset Item.

    https://huggingface.co/datasets/NeelNanda/c4-tokenized-2b
    """

    tokens: list[list[int]]


@final
class NeelC4SourceDataset(SourceDataset[NeelC4SourceDataBatch]):
    """Neel Nanda C4 Pre-Tokenized 2B Dataset.

    https://huggingface.co/datasets/monology/pile-uncopyrighted
    """

    def preprocess(
        self,
        source_batch: NeelC4SourceDataBatch,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts.

        As this dataset is already tokenized, all this does is split up each item based on the
        context size.

        Args:
            source_batch: A batch of source data.
            context_size: The context size to use when returning a list of tokenized prompts.
        """
        tokenized_prompts: list[list[int]] = source_batch["tokens"]

        # Chunk each tokenized prompt into blocks of context_size, discarding the last block if too
        # small.
        context_size_prompts = []
        for encoding in tokenized_prompts:
            chunks = [
                encoding[i : i + context_size]
                for i in range(0, len(encoding), context_size)
                if len(encoding[i : i + context_size]) == context_size
            ]
            context_size_prompts.extend(chunks)

        return {"input_ids": context_size_prompts}

    def __init__(
        self,
        context_size: int = 250,
        buffer_size: int = 1000,
        preprocess_batch_size: int = 1000,
        dataset_path: str = "NeelNanda/c4-tokenized-2b",
        dataset_split: str = "train",
    ):
        """Initialize the Pile Uncopyrighted dataset.

        Example:
            >>> data = NeelC4SourceDataset()
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
        super().__init__(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            context_size=context_size,
            buffer_size=buffer_size,
            preprocess_batch_size=preprocess_batch_size,
        )
