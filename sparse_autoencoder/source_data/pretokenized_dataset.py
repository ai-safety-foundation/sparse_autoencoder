"""Pre-Tokenized Dataset from Hugging Face.

PreTokenizedDataset should work with any of the following tokenized datasets:
- NeelNanda/pile-small-tokenized-2b
- NeelNanda/pile-tokenized-10b
- NeelNanda/openwebtext-tokenized-9b
- NeelNanda/c4-tokenized-2b
- NeelNanda/code-tokenized
- NeelNanda/c4-code-tokenized-2b
- NeelNanda/pile-old-tokenized-2b
- alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2

"""
from collections.abc import Mapping, Sequence
from typing import final

from pydantic import PositiveInt, validate_call

from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TokenizedPrompts


@final
class PreTokenizedDataset(SourceDataset[dict]):
    """General Pre-Tokenized Dataset from Hugging Face.

    Can be used for various datasets available on Hugging Face.
    """

    def preprocess(
        self,
        source_batch: dict,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts.

        The method splits each pre-tokenized item based on the context size.

        Args:
            source_batch: A batch of source data.
            context_size: The context size to use for tokenized prompts.

        Returns:
            Tokenized prompts.

        Raises:
            ValueError: If the context size is larger than the tokenized prompt size.
        """
        tokenized_prompts: list[list[int]] = source_batch[self._dataset_column_name]

        # Check the context size is not too large
        if context_size > len(tokenized_prompts[0]):
            error_message = (
                f"The context size ({context_size}) is larger than the "
                f"tokenized prompt size ({len(tokenized_prompts[0])})."
            )
            raise ValueError(error_message)

        # Chunk each tokenized prompt into blocks of context_size,
        # discarding the last block if too small.
        context_size_prompts = []
        for encoding in tokenized_prompts:
            chunks = [
                encoding[i : i + context_size]
                for i in range(0, len(encoding), context_size)
                if len(encoding[i : i + context_size]) == context_size
            ]
            context_size_prompts.extend(chunks)

        return {"input_ids": context_size_prompts}

    @validate_call
    def __init__(
        self,
        dataset_path: str,
        context_size: PositiveInt = 256,
        buffer_size: PositiveInt = 1000,
        dataset_dir: str | None = None,
        dataset_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        dataset_split: str = "train",
        dataset_column_name: str = "input_ids",
        preprocess_batch_size: PositiveInt = 1000,
        *,
        pre_download: bool = False,
    ):
        """Initialize a pre-tokenized dataset from Hugging Face.

        Args:
            dataset_path: The path to the dataset on Hugging Face (e.g.
                `alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2).
            context_size: The context size for tokenized prompts.
            buffer_size: The buffer size to use when shuffling the dataset when streaming. When
                streaming a dataset, this just pre-downloads at least `buffer_size` items and then
                shuffles just that buffer. Note that the generated activations should also be
                shuffled before training the sparse autoencoder, so a large buffer may not be
                strictly necessary here. Note also that this is the number of items in the dataset
                (e.g. number of prompts) and is typically significantly less than the number of
                tokenized prompts once the preprocessing function has been applied.
            dataset_dir: Defining the `data_dir` of the dataset configuration.
            dataset_files: Path(s) to source data file(s).
            dataset_split: Dataset split (e.g. `train`).
            dataset_column_name: The column name for the tokenized prompts.
            preprocess_batch_size: The batch size to use just for preprocessing the dataset (e.g.
                tokenizing prompts).
            pre_download: Whether to pre-download the whole dataset.
        """
        super().__init__(
            buffer_size=buffer_size,
            context_size=context_size,
            dataset_dir=dataset_dir,
            dataset_files=dataset_files,
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            dataset_column_name=dataset_column_name,
            pre_download=pre_download,
            preprocess_batch_size=preprocess_batch_size,
        )
