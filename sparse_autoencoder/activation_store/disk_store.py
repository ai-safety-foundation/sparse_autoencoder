"""Disk Activation Store."""
from pathlib import Path
import re
import tempfile

from jaxtyping import Float
import torch
from torch import Tensor

from sparse_autoencoder.activation_store.base_store import (
    ActivationStore,
)
from sparse_autoencoder.tensor_types import Axis


DEFAULT_DISK_ACTIVATION_STORE_PATH = Path(tempfile.gettempdir()) / "activation_store"


class DiskActivationStore(ActivationStore):
    """Disk Activation Store.

    Stores activation vectors on disk (hard-drive). Makes use of a queue (buffer) to store up
    activation vectors and then write them to the disk in batches.

    Multiprocess safe (supports writing from multiple GPU workers).

    Warning:
    Unless you want to keep and use existing .pt files in the storage directory when initialized,
    set `empty_dir` to `True`.

    Note also that :meth:`close` must be called to ensure all activation vectors are written to disk
    after the last batch has been added to the store.
    """

    _storage_path: Path
    """Path to the Directory where the Activation Vectors are Stored."""

    _cache: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)]
    """Cache for Activation Vectors.

    Activation vectors are buffered in memory until the cache is full, at which point they are
    written to disk.
    """

    _cache_device: torch.device = torch.device("cpu")
    """Cache device."""

    _items_stored: list[int]
    """Number of items stored."""

    _max_cache_size: int
    """Maximum Number of Activation Vectors to cache in Memory."""

    _disk_n_activation_vectors_per_component: int | None
    """Length of the Store (on disk)."""

    _num_components: int
    """Number of components"""

    @property
    def num_components(self) -> int:
        """Number of components."""
        return self._num_components

    @property
    def current_activations_stored_per_component(self) -> list[int]:
        """Current activations stored per component."""
        disk_items_stored = len(self)
        return [cache_items + disk_items_stored for cache_items in self._items_stored]

    def __init__(
        self,
        num_neurons: int,
        storage_path: Path = DEFAULT_DISK_ACTIVATION_STORE_PATH,
        max_cache_size: int = 10_000,
        num_components: int = 1,
        *,
        empty_dir: bool = False,
    ):
        """Initialize the Disk Activation Store.

        Args:
            num_neurons: Number of neurons in each activation vector.
            storage_path: Path to the directory where the activation vectors will be stored.
            max_cache_size: The maximum number of activation vectors (per component) to cache in
                memory before writing to disk.
            num_components: Number of components to store (i.e. number of source models).
            empty_dir: Whether to empty the directory before writing. Generally you want to set this
                to `True` as otherwise the directory may contain stale activation vectors from
                previous runs. However if you are just initialising a pre-created store, set it as
                False.
        """
        super().__init__()

        self._max_cache_size = max_cache_size
        self._num_components = num_components

        # Setup the storage directory
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        if empty_dir:
            self.empty()

        # Setup the Cache
        self._cache = torch.empty(
            (max_cache_size, num_components, num_neurons), device=self._cache_device
        )
        self._items_stored = [0] * num_components

    def _write_to_disk(self) -> None:
        """Write the contents of the cache to disk.

        Example:
            >>> store = DiskActivationStore(max_cache_size=2, empty_dir=True, num_neurons=100)
            >>> store.append(torch.randn(100))
            >>> store._write_to_disk()
            >>> print(len(store))
            1
        """
        # Save to disk
        items = self._cache[: min(self._items_stored)]
        filename = f"{len(self)}-{self._items_stored}.pt"
        torch.save(items, self._storage_path / filename)

        # Update the number of items stored
        self._disk_n_activation_vectors_per_component = min(self._items_stored) + (
            self._disk_n_activation_vectors_per_component or 0
        )

        # Empty the cache (note we just need to mark as empty so we can start filling it again)
        self._items_stored = [0] * self._num_components

    def append(
        self,
        item: Float[Tensor, (Axis.INPUT_OUTPUT_FEATURE)],
        component_idx: int = 0,
    ) -> None:
        """Add a Single Item to the Store.

        Example:
        >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True, num_neurons=100)
        >>> store.append(torch.randn(100))
        >>> store.append(torch.randn(100)) # Triggers a write of the last item to disk
        >>> print(len(store))
        1

        Args:
            item: Activation vector to add to the store.
            component_idx: The component index to append the item to.
        """
        # Write to disk first if full (note this also resets items stored)
        if self._items_stored[component_idx] + 1 > self._max_cache_size:
            self._write_to_disk()

        # Add to cache
        self._cache[self._items_stored[component_idx]] = item.to(self._cache_device)
        self._items_stored[component_idx] += 1

    def extend(
        self,
        batch: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
        component_idx: int = 1,
    ) -> None:
        """Add a Batch to the Store.

        Example:
            >>> store = DiskActivationStore(max_cache_size=10, empty_dir=True, num_neurons=100)
            >>> store.extend(torch.randn(10, 100))
            >>> store.extend(torch.randn(10, 100)) # Triggers a write of the last items to disk
            >>> print(len(store))
            10

        Args:
            batch: Batch of activation vectors to add to the store.
            component_idx: The component index to append the item to.

        Raises:
            ValueError: If the batch is larger than the cache size.
        """
        num_activation_tensors: int = batch.shape[0]

        # Check the batch is smaller than the cache size
        if num_activation_tensors > self._max_cache_size:
            error_message = (
                f"Batch size {num_activation_tensors} is larger than the cache size "
                f"{self._max_cache_size}."
            )
            raise ValueError(error_message)

        # Write to disk first if full (note this also resets items stored)
        if self._items_stored[component_idx] + num_activation_tensors > self._max_cache_size:
            self._write_to_disk()

        # Add to cache
        self._cache[
            self._items_stored[component_idx] : self._items_stored[component_idx]
            + num_activation_tensors,
            component_idx,
        ] = batch.to(self._cache_device)

        self._items_stored[component_idx] += num_activation_tensors

    def wait_for_writes_to_complete(self) -> None:
        """Wait for Writes to Complete.

        This should be called after the last batch has been added to the store. It will wait for
        all activation vectors to be written to disk.

        Example:
            >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True, num_neurons=100)
            >>> store.append(torch.randn(100))
            >>> store.wait_for_writes_to_complete()
            >>> print(len(store))
            1
        """
        if min(self._items_stored) > 0:
            self._write_to_disk()

    @property
    def _all_filenames(self) -> list[Path]:
        """Return a List of All Activation Vector Filenames."""
        return list(self._storage_path.glob("*.pt"))

    def empty(self) -> None:
        """Empty the Store.

        Warning:
        This will delete all .pt files in the top level of the storage directory.

        Example:
            >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True, num_neurons=100)
            >>> store.append(torch.randn(100))
            >>> store.append(torch.randn(100))
            >>> print(len(store))
            1

            >>> store.empty()
            >>> print(len(store))
            0
        """
        for file in self._all_filenames:
            file.unlink()
        self._disk_n_activation_vectors_per_component = 0

    @staticmethod
    def get_idx_from_filename(filename: Path) -> tuple[int, int]:
        """Get the end index from a filename.

        Example:
            >>> filename = Path("0-100.pt")
            >>> DiskActivationStore.get_idx_from_filename(filename)
            (0, 100)

        Args:
            filename: Filename to extract the end index from.

        Returns:
            The end index of the filename.
        """
        numbers = re.findall(r"\d+", str(filename))
        return (numbers[0], numbers[1])

    def __getitem__(
        self, index: int
    ) -> Float[Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]:
        """Get Item Dunder Method.

        Warning:
            This is very inefficient and should only be used for testing. For training, consider
                using a DataLoader that iterates over the disk store directory.

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        # Find the file containing the activation vector
        for filename in self._all_filenames:
            indexes = self.get_idx_from_filename(filename)

            # Load if the index is in the range of the file
            if index > indexes[0] and index < indexes[1]:
                activation_vectors = torch.load(filename)
                return activation_vectors[index % self._max_cache_size]

        # If still not found
        error_message = f"Index {index} out of range."
        raise IndexError(error_message)

    def __len__(self) -> int:
        """Length Dunder Method.

        Example:
            >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True, num_neurons=100)
            >>> print(len(store))
            0

        Returns:
            The number of activation vectors in the dataset.
        """
        # Calculate the length if not cached
        if self._disk_n_activation_vectors_per_component is None:
            max_size: int = 0

            for filename in self._all_filenames:
                filename_end_idx = self.get_idx_from_filename(filename)[1]
                self._disk_n_activation_vectors_per_component = max(max_size, filename_end_idx)

            self._disk_n_activation_vectors_per_component = max_size

        return self._disk_n_activation_vectors_per_component

    def __del__(self) -> None:
        """Delete Dunder Method."""
        self.wait_for_writes_to_complete()
