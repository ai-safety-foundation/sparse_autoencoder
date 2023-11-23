"""Disk Activation Store."""
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Manager
from multiprocessing.managers import ListProxy, ValueProxy
from pathlib import Path
import tempfile
from threading import Lock

import torch

from sparse_autoencoder.activation_store.base_store import (
    ActivationStore,
)
from sparse_autoencoder.activation_store.utils.extend_resize import (
    resize_to_list_vectors,
)
from sparse_autoencoder.tensor_types import (
    InputOutputActivationVector,
    SourceModelActivations,
)


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

    _cache: ListProxy
    """Cache for Activation Vectors.

    Activation vectors are buffered in memory until the cache is full, at which point they are
    written to disk.
    """

    _cache_lock: Lock
    """Lock for the Cache."""

    _max_cache_size: int
    """Maximum Number of Activation Vectors to cache in Memory."""

    _thread_pool: ThreadPoolExecutor
    """Threadpool for non-blocking writes to the file system."""

    _disk_n_activation_vectors: ValueProxy[int]
    """Length of the Store (on disk).

    Minus 1 signifies not calculated yet.
    """

    def __init__(
        self,
        storage_path: Path = DEFAULT_DISK_ACTIVATION_STORE_PATH,
        max_cache_size: int = 10_000,
        num_workers: int = 6,
        *,
        empty_dir: bool = False,
    ):
        """Initialize the Disk Activation Store.

        Args:
            storage_path: Path to the directory where the activation vectors will be stored.
            max_cache_size: The maximum number of activation vectors to cache in memory before
                writing to disk. Note this is only followed approximately.
            num_workers: Number of CPU workers to use for non-blocking writes to the file system (so
                that the model can keep running whilst it writes the previous activations to disk).
                This should be less than the number of CPU cores available. You don't need multiple
                GPUs to take advantage of this feature.
            empty_dir: Whether to empty the directory before writing. Generally you want to set this
                to `True` as otherwise the directory may contain stale activation vectors from
                previous runs.
        """
        super().__init__()

        # Setup the storage directory
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Setup the Cache
        manager = Manager()
        self._cache = manager.list()
        self._max_cache_size = max_cache_size
        self._cache_lock = manager.Lock()
        self._disk_n_activation_vectors = manager.Value("i", -1)

        # Empty the directory if needed
        if empty_dir:
            self.empty()

        # Create a threadpool for non-blocking writes to the cache
        self._thread_pool = ThreadPoolExecutor(num_workers)

    def _write_to_disk(self, *, wait_for_max: bool = False) -> None:
        """Write the contents of the queue to disk.

        Args:
            wait_for_max: Whether to wait until the cache is full before writing to disk.
        """
        with self._cache_lock:
            # Check we have enough items
            if len(self._cache) == 0:
                return

            size_to_get = min(self._max_cache_size, len(self._cache))
            if wait_for_max and size_to_get < self._max_cache_size:
                return

            # Get the activations from the cache and delete them
            activations = self._cache[0:size_to_get]
            del self._cache[0:size_to_get]

            # Update the length cache
            if self._disk_n_activation_vectors.value != -1:
                self._disk_n_activation_vectors.value += len(activations)

        stacked_activations = torch.stack(activations)

        filename = f"{self.__len__}.pt"
        torch.save(stacked_activations, self._storage_path / filename)

    def append(self, item: InputOutputActivationVector) -> Future | None:
        """Add a Single Item to the Store.

        Example:
        >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True)
        >>> future = store.append(torch.randn(100))
        >>> future.result()
        >>> print(len(store))
        1

        Args:
            item: Activation vector to add to the store.

        Returns:
            Future that completes when the activation vector has queued to be written to disk, and
            if needed, written to disk.
        """
        with self._cache_lock:
            self._cache.append(item)

            # Write to disk if needed
            if len(self._cache) >= self._max_cache_size:
                return self._thread_pool.submit(self._write_to_disk, wait_for_max=True)

        return None  # Keep mypy happy

    def extend(self, batch: SourceModelActivations) -> Future | None:
        """Add a Batch to the Store.

        Example:
        >>> store = DiskActivationStore(max_cache_size=10, empty_dir=True)
        >>> future = store.extend(torch.randn(10, 100))
        >>> future.result()
        >>> print(len(store))
        10

        Args:
            batch: Batch of activation vectors to add to the store.

        Returns:
            Future that completes when the activation vectors have queued to be written to disk, and
            if needed, written to disk.
        """
        items: list[InputOutputActivationVector] = resize_to_list_vectors(batch)

        with self._cache_lock:
            self._cache.extend(items)

            # Write to disk if needed
            if len(self._cache) >= self._max_cache_size:
                return self._thread_pool.submit(self._write_to_disk, wait_for_max=True)

        return None  # Keep mypy happy

    def wait_for_writes_to_complete(self) -> None:
        """Wait for Writes to Complete.

        This should be called after the last batch has been added to the store. It will wait for
        all activation vectors to be written to disk.

        Example:
        >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True)
        >>> future = store.append(torch.randn(100))
        >>> store.wait_for_writes_to_complete()
        >>> print(len(store))
        1
        """
        while len(self._cache) > 0:
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
        >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True)
        >>> future = store.append(torch.randn(100))
        >>> future.result()
        >>> print(len(store))
        1

        >>> store.empty()
        >>> print(len(store))
        0
        """
        for file in self._all_filenames:
            file.unlink()
        self._disk_n_activation_vectors.value = 0

    def __getitem__(self, index: int) -> InputOutputActivationVector:
        """Get Item Dunder Method.

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.
        """
        # Find the file containing the activation vector
        file_index = index // self._max_cache_size
        file = self._storage_path / f"{file_index}.pt"

        # Load the file and return the activation vector
        activation_vectors = torch.load(file)
        return activation_vectors[index % self._max_cache_size]

    def __len__(self) -> int:
        """Length Dunder Method.

        Example:
            >>> store = DiskActivationStore(max_cache_size=1, empty_dir=True)
            >>> print(len(store))
            0

        Returns:
            The number of activation vectors in the dataset.
        """
        # Calculate the length if not cached
        if self._disk_n_activation_vectors.value == -1:
            cache_size: int = 0
            for file in self._all_filenames:
                cache_size += len(torch.load(file))
            self._disk_n_activation_vectors.value = cache_size

        return self._disk_n_activation_vectors.value

    def __del__(self) -> None:
        """Delete Dunder Method."""
        # Shutdown the thread pool after everything is complete
        self._thread_pool.shutdown(wait=True, cancel_futures=False)
        self.wait_for_writes_to_complete()
