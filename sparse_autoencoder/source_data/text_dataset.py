"""Generic Text Dataset Module for Hugging Face Datasets.

GenericTextDataset should work with the following datasets:
- monology/pile-uncopyrighted
- the_pile_openwebtext2
- roneneldan/TinyStories-33M
- roneneldan/TinyStories-8M
- roneneldan/TinyStories-3M
- roneneldan/TinyStories-1Layer-21M
- roneneldan/TinyStories-1M
- roneneldan/TinyStories-2Layers-33M
- roneneldan/TinyStories-Instruct-2Layers-33M
- roneneldan/TinyStories-Instruct-28M
- roneneldan/TinyStories-Instruct-33M
- roneneldan/TinyStories-Instruct-8M
- roneneldan/TinyStories-Instruct-3M
- roneneldan/TinyStories-Instruct-1M
- roneneldan/TinyStories-Instuct-1Layer-21M
- roneneldan/TinyStories-28M
"""
from typing import TypedDict, final

from transformers import PreTrainedTokenizerBase

from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TokenizedPrompts


class GenericTextDataBatch(TypedDict):
    """Generic Text Dataset Batch.

    Assumes the dataset provides a 'text' field with a list of strings.
    """

    text: list[str]
    meta: list[dict[str, dict[str, str]]]  # Optional, depending on the dataset structure.


@final
class GenericTextDataset(SourceDataset[GenericTextDataBatch]):
    """Generic Text Dataset for any text-based dataset from Hugging Face."""

    tokenizer: PreTrainedTokenizerBase

    def preprocess(
        self,
        source_batch: GenericTextDataBatch,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts.

        Tokenizes and chunks text data into lists of tokenized prompts with specified context size.

        Args:
            source_batch: A batch of source data, including 'text' with a list of strings.
            context_size: Context size for tokenized prompts.

        Returns:
            Tokenized prompts.
        """
        prompts: list[str] = source_batch["text"]

        tokenized_prompts = self.tokenizer(prompts, truncation=True, padding=False)

        # Chunk each tokenized prompt into blocks of context_size, discarding incomplete blocks.
        context_size_prompts = []
        for encoding in list(tokenized_prompts["input_ids"]):  # type: ignore
            chunks = [
                encoding[i : i + context_size]
                for i in range(0, len(encoding), context_size)
                if len(encoding[i : i + context_size]) == context_size
            ]
            context_size_prompts.extend(chunks)

        return {"input_ids": context_size_prompts}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_size: int = 250,
        buffer_size: int = 1000,
        preprocess_batch_size: int = 1000,
        dataset_path: str = "monology/pile-uncopyrighted",
        dataset_split: str = "train",
    ):
        """Initialize a generic text dataset from Hugging Face.

        Args:
            tokenizer: Tokenizer to process text data.
            context_size: Context size for tokenized prompts.
            buffer_size: Buffer size for shuffling the dataset.
            preprocess_batch_size: Batch size for preprocessing.
            dataset_path: Path to the dataset on Hugging Face.
            dataset_split: Dataset split (e.g., 'train').
        """
        self.tokenizer = tokenizer

        super().__init__(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            context_size=context_size,
            buffer_size=buffer_size,
            preprocess_batch_size=preprocess_batch_size,
        )
