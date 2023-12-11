"""Generic Text Dataset Module for Hugging Face Datasets.

GenericTextDataset should work with the following datasets:
- monology/pile-uncopyrighted
- the_pile_openwebtext2
- roneneldan/TinyStories
"""
from collections.abc import Mapping, Sequence
from typing import TypedDict, final

from datasets import IterableDataset
from transformers import PreTrainedTokenizerBase

from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TokenizedPrompts


class GenericTextDataBatch(TypedDict):
    """Generic Text Dataset Batch.

    Assumes the dataset provides a 'text' field with a list of strings.
    """

    text: list[str]
    meta: list[dict[str, dict[str, str]]]  # Optional, depending on the dataset structure.


@final
class TextDataset(SourceDataset[GenericTextDataBatch]):
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
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        buffer_size: int = 1000,
        context_size: int = 256,
        dataset_dir: str | None = None,
        dataset_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        dataset_split: str = "train",
        n_processes_preprocessing: int | None = None,
        preprocess_batch_size: int = 1000,
        *,
        pre_download: bool = False,
    ):
        """Initialize a generic text dataset from Hugging Face.

        Args:
            dataset_path: Path to the dataset on Hugging Face (e.g. `'monology/pile-uncopyright'`).
            tokenizer: Tokenizer to process text data.
            buffer_size: The buffer size to use when shuffling the dataset when streaming. When
                streaming a dataset, this just pre-downloads at least `buffer_size` items and then
                shuffles just that buffer. Note that the generated activations should also be
                shuffled before training the sparse autoencoder, so a large buffer may not be
                strictly necessary here. Note also that this is the number of items in the dataset
                (e.g. number of prompts) and is typically significantly less than the number of
                tokenized prompts once the preprocessing function has been applied.
            context_size: The context size to use when returning a list of tokenized prompts.
                *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
                a context size of 250.
            dataset_dir: Defining the `data_dir` of the dataset configuration.
            dataset_files: Path(s) to source data file(s).
            dataset_split: Dataset split (e.g., 'train').
            n_processes_preprocessing: Number of processes to use for preprocessing.
            preprocess_batch_size: Batch size for preprocessing (tokenizing prompts).
            pre_download: Whether to pre-download the whole dataset.
        """
        self.tokenizer = tokenizer

        super().__init__(
            buffer_size=buffer_size,
            context_size=context_size,
            dataset_dir=dataset_dir,
            dataset_files=dataset_files,
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            n_processes_preprocessing=n_processes_preprocessing,
            pre_download=pre_download,
            preprocess_batch_size=preprocess_batch_size,
        )

    def push_to_hugging_face_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload preprocessed dataset using sparse_autoencoder.",
        max_shard_size: str | None = None,
        num_shards: int = 64,
        revision: str = "main",
        *,
        private: bool = False,
    ) -> None:
        """Share preprocessed dataset to Hugging Face hub.

        Motivation:
            Pre-processing a dataset can be time-consuming, so it is useful to be able to share the
            pre-processed dataset with others. This function allows you to do that by pushing the
            pre-processed dataset to the Hugging Face hub.

        Warning:
            You must be logged into HuggingFace (e.g with `huggingface-cli login` from the terminal)
            to use this.

        Warning:
            This will only work if the dataset is not streamed (i.e. if `pre_download=True` when
            initializing the dataset).

        Args:
            repo_id: Hugging Face repo ID to save the dataset to (e.g. `username/dataset_name`).
            commit_message: Commit message.
            max_shard_size: Maximum shard size (e.g. `'500MB'`). Should not be set if `num_shards`
                is set.
            num_shards: Number of shards to split the dataset into. A high number is recommended
                here to allow for flexible distributed training of SAEs across nodes (where e.g.
                each node fetches it's own shard).
            revision: Branch to push to.
            private: Whether to save the dataset privately.

        Raises:
            TypeError: If the dataset is streamed.
        """
        if isinstance(self.dataset, IterableDataset):
            error_message = (
                "Cannot share a streamed dataset to Hugging Face. "
                "Please use `pre_download=True` when initializing the dataset."
            )
            raise TypeError(error_message)

        self.dataset.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            max_shard_size=max_shard_size,
            num_shards=num_shards,
            private=private,
            revision=revision,
        )
