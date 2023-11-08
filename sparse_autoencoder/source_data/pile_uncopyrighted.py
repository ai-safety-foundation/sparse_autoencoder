"""The Pile Uncopyrighted Dataset."""
from typing import TypedDict, final

from transformers import PreTrainedTokenizerBase

from sparse_autoencoder.source_data.abstract_dataset import (
    SourceDataset,
    TokenizedPrompts,
)


class PileUncopyrightedSourceDataBatch(TypedDict):
    """Pile Uncopyrighted Source Data.

    https://huggingface.co/datasets/monology/pile-uncopyright
    """

    text: list[str]
    meta: list[dict[str, dict[str, str]]]


@final
class PileUncopyrightedDataset(SourceDataset[PileUncopyrightedSourceDataBatch]):
    """The Pile Uncopyrighted Dataset.

    https://huggingface.co/datasets/monology/pile-uncopyrighted
    """

    tokenizer: PreTrainedTokenizerBase

    def preprocess(
        self,
        source_batch: PileUncopyrightedSourceDataBatch,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts.

        For each prompt's `text`, tokenize it and chunk into a list of tokenized prompts of length
        `context_size`. For the last item in the chunk, throw it away if the length is less than
        `context_size` (i.e. if it would otherwise require padding). Then finally flatten all
        batches to a single list of tokenized prompts.

        Args:
            source_batch: A batch of source data. For example, with The Pile dataset this would be a
                dict including the key "text" with a value of a list of strings (not yet tokenized).
            context_size: The context size to use when returning a list of tokenized prompts.
        """
        prompts: list[str] = source_batch["text"]

        tokenized_prompts = self.tokenizer(prompts)

        # Chunk each tokenized prompt into blocks of context_size, discarding the last block if too
        # small.
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
        """Initialize the Pile Uncopyrighted dataset.

        Example:
            >>> from transformers import GPT2TokenizerFast
            >>> tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            >>> data = PileUncopyrightedDataset(
            ...     tokenizer=tokenizer
            ... )
            >>> first_item = next(iter(data))
            >>> len(first_item["input_ids"])
            250

        Args:
            tokenizer: The tokenizer to use to tokenize the prompts.
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
        self.tokenizer = tokenizer

        super().__init__(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            context_size=context_size,
            buffer_size=buffer_size,
            preprocess_batch_size=preprocess_batch_size,
        )
