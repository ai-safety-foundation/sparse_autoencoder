"""Abstract tokenized prompts dataset class."""
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Generic, TypedDict, TypeVar, final

from datasets import Dataset, IterableDataset, load_dataset
from jaxtyping import Int
from pydantic import PositiveInt, validate_call
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from sparse_autoencoder.tensor_types import Axis


TokenizedPrompt = list[int]
"""A tokenized prompt."""


class TokenizedPrompts(TypedDict):
    """Tokenized prompts."""

    input_ids: list[TokenizedPrompt]


class TorchTokenizedPrompts(TypedDict):
    """Tokenized prompts prepared for PyTorch."""

    input_ids: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)]


HuggingFaceDatasetItem = TypeVar("HuggingFaceDatasetItem", bound=Any)
"""Hugging face dataset item typed dict.

When extending :class:`SourceDataset` you should create a `TypedDict` that matches the structure of
each dataset item in the underlying Hugging Face dataset.

Example:
    With the [Uncopyrighted
    Pile](https://huggingface.co/datasets/monology/pile-uncopyrighted) this should be a typed dict
    with text and meta properties.

    >>> class PileUncopyrightedSourceDataBatch(TypedDict):
    ...    text: list[str]
    ...    meta: list[dict[str, dict[str, str]]]
"""


class SourceDataset(ABC, Generic[HuggingFaceDatasetItem]):
    """Abstract source dataset.

    Source dataset that is used to generate the activations dataset (by running forward passes of
    the source model with this data). It should contain prompts that have been tokenized with no
    padding tokens (apart from an optional single first padding token). This enables efficient
    generation of the activations dataset.

    Wraps an HuggingFace IterableDataset.
    """

    context_size: int
    """Number of tokens in the context window.

    The paper *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
    a context size of 250.
    """

    dataset: Dataset | IterableDataset
    """Underlying HuggingFace Dataset.

    Warning:
        Hugging Face `Dataset` objects are confusingly not the same as PyTorch `Dataset` objects.
    """

    _dataset_column_name: str
    """Dataset column name for the prompts."""

    @abstractmethod
    def preprocess(
        self,
        source_batch: HuggingFaceDatasetItem,
        *,
        context_size: int,
    ) -> TokenizedPrompts:
        """Preprocess function.

        Takes a `preprocess_batch_size` ($m$) batch of source data (which may e.g. include string
        prompts), and returns a dict with a single key of `input_ids` and a value of an arbitrary
        length list ($n$) of tokenized prompts. Note that $m$ does not have to be equal to $n$.

        Applied to the dataset with the [Hugging Face
        Dataset](https://huggingface.co/docs/datasets/v2.14.5/en/package_reference/main_classes#datasets.Dataset.map)
        `map` function.

        Warning:
            The returned tokenized prompts should not have any padding tokens (apart from an
            optional single first padding token).

        Args:
            source_batch: A batch of source data. For example, with The Pile dataset this would be a
                dict including the key "text" with a value of a list of strings (not yet tokenized).
            context_size: The context size to use when returning a list of tokenized prompts.
                *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
                a context size of 250.

        Returns:
            Tokenized prompts.
        """

    @abstractmethod
    @validate_call
    def __init__(
        self,
        dataset_path: str,
        dataset_split: str,
        context_size: PositiveInt,
        buffer_size: PositiveInt = 1000,
        dataset_dir: str | None = None,
        dataset_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        dataset_column_name: str = "input_ids",
        n_processes_preprocessing: PositiveInt | None = None,
        preprocess_batch_size: PositiveInt = 1000,
        *,
        pre_download: bool = False,
    ):
        """Initialise the dataset.

        Loads the dataset with streaming from HuggingFace, dds preprocessing and shuffling to the
        underlying Hugging Face `IterableDataset`.

        Args:
            dataset_path: The path to the dataset on Hugging Face.
            dataset_split: Dataset split (e.g. `train`).
            context_size: The context size to use when returning a list of tokenized prompts.
                *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning* used
                a context size of 250.
            buffer_size: The buffer size to use when shuffling the dataset when streaming. When
                streaming a dataset, this just pre-downloads at least `buffer_size` items and then
                shuffles just that buffer. Note that the generated activations should also be
                shuffled before training the sparse autoencoder, so a large buffer may not be
                strictly necessary here. Note also that this is the number of items in the dataset
                (e.g. number of prompts) and is typically significantly less than the number of
                tokenized prompts once the preprocessing function has been applied.
            dataset_dir: Defining the `data_dir` of the dataset configuration.
            dataset_files: Path(s) to source data file(s).
            dataset_column_name: The column name for the prompts.
            n_processes_preprocessing: The number of processes to use for preprocessing.
            preprocess_batch_size: The batch size to use just for preprocessing the dataset (e.g.
                tokenizing prompts).
            pre_download: Whether to pre-download the whole dataset.

        Raises:
            TypeError: If the loaded dataset is not a Hugging Face `Dataset` or `IterableDataset`.
        """
        self.context_size = context_size
        self._dataset_column_name = dataset_column_name

        # Load the dataset
        should_stream = not pre_download
        dataset = load_dataset(
            dataset_path,
            streaming=should_stream,
            split=dataset_split,
            data_dir=dataset_dir,
            data_files=dataset_files,
        )

        # Setup preprocessing (we remove all columns except for input ids)
        remove_columns: list[str] = list(next(iter(dataset)).keys())
        if "input_ids" in remove_columns:
            remove_columns.remove("input_ids")

        if pre_download:
            if not isinstance(dataset, Dataset):
                error_message = (
                    f"Expected Hugging Face dataset to be a Dataset when pre-downloading, but got "
                    f"{type(dataset)}."
                )
                raise TypeError(error_message)

            # Download the whole dataset
            mapped_dataset = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=preprocess_batch_size,
                fn_kwargs={"context_size": context_size},
                remove_columns=remove_columns,
                num_proc=n_processes_preprocessing,
            )
            self.dataset = mapped_dataset.shuffle()
        else:
            # Setup approximate shuffling. As the dataset is streamed, this just pre-downloads at
            # least `buffer_size` items and then shuffles just that buffer.
            # https://huggingface.co/docs/datasets/v2.14.5/stream#shuffle
            if not isinstance(dataset, IterableDataset):
                error_message = (
                    f"Expected Hugging Face dataset to be an IterableDataset when streaming, but "
                    f"got {type(dataset)}."
                )
                raise TypeError(error_message)

            mapped_dataset = dataset.map(
                self.preprocess,
                batched=True,
                batch_size=preprocess_batch_size,
                fn_kwargs={"context_size": context_size},
                remove_columns=remove_columns,
            )
            self.dataset = mapped_dataset.shuffle(buffer_size=buffer_size)  # type: ignore

    @final
    def __iter__(self) -> Any:  # noqa: ANN401
        """Iterate Dunder Method.

        Enables direct access to :attr:`dataset` with e.g. `for` loops.
        """
        return self.dataset.__iter__()

    @final
    def get_dataloader(self, batch_size: int) -> DataLoader[TorchTokenizedPrompts]:
        """Get a PyTorch DataLoader.

        Args:
            batch_size: The batch size to use.

        Returns:
            PyTorch DataLoader.
        """
        torch_dataset: TorchDataset[TorchTokenizedPrompts] = self.dataset.with_format("torch")  # type: ignore

        return DataLoader[TorchTokenizedPrompts](
            torch_dataset,
            batch_size=batch_size,
            # Shuffle is most efficiently done with the `shuffle` method on the dataset itself, not
            # here.
            shuffle=False,
        )
