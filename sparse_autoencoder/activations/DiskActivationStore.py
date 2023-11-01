"""Disk Activation Store."""
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Lock, Queue, Value
from multiprocessing.sharedctypes import Synchronized, SynchronizedBase
from pathlib import Path

import torch

from sparse_autoencoder.activations.ActivationStore import (
    ActivationStore,
    ActivationStoreBatch,
    ActivationStoreItem,
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

    Args:
        storage_path: Path to the directory where the activation vectors will be stored. Defaults to
            the OS temporary directory.
        empty_dir: Whether to empty the directory before writing. Generally you want to set this to
            `True` as otherwise the directory may contain stale activation vectors from previous
            runs.
        max_queue_size: The maximum number of activation vectors to buffer in memory before writing
            to disk.
        num_workers: Number of CPU workers to use for non-blocking writes to the file system (so that
            the model can keep running whilst it writes the previous activations to disk). This
            should be less than the number of CPU cores available. You don't need multiple GPUs to
            take advantage of this feature.
    """

    _storage_path: Path
    """Path to the Directory where the Activation Vectors are Stored."""

    _queue: Queue
    """Queue to Buffer Activation Vectors in Memory.
    
    Activation vectors are buffered in memory until the queue is full, at which point they are
    written to disk.
    """

    _queue_size: Synchronized
    """Number of Activation Vectors in the Queue."""

    _max_queue_size: int
    """Maximum Number of Activation Vectors to Buffer in Memory.
    
    Note this is approximately respected, as multiprocessing doesn't have strict guarantees.
    """

    length_cache: int | None = None
    """Cache of the Length of the Store.
    
    This is cached as it's expensive to calculate.
    """

    _thread_pool: ThreadPoolExecutor
    """Threadpool for non-blocking writes to the file system."""

    def __init__(
        self,
        storage_path: Path = DEFAULT_DISK_ACTIVATION_STORE_PATH,
        empty_dir: bool = False,
        max_queue_size: int = 10_000,
        num_workers: int = 6,
    ):
        super().__init__()

        # Setup the storage directory
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        if empty_dir:
            self.empty()
            self.length_cache = 0

        # Setup the Queue
        # Note we don't explicitly set the maxsize of the queue, as we want to allow it to grow
        # beyond the max_queue_size if the workers are slow to write to disk (as this will block
        # the model from running otherwise).
        self._queue = Queue()
        self._queue_size: Synchronized = Value("i", 0)  # type: ignore
        self._queue_size_lock = Lock()
        self._max_queue_size = max_queue_size

        # Create a threadpool for non-blocking writes to the dataset
        self._thread_pool = ThreadPoolExecutor(num_workers)

    def _write_to_disk(self) -> None:
        """Write the contents of the queue to disk."""
        all_queue_items = []
        for _ in range(self._queue_size.value):
            item = self._queue.get(block=True)
            all_queue_items.append(item)
            with self._queue_size_lock:
                self._queue_size.value -= 1

        stacked_activation_vectors = torch.stack(all_queue_items)
        filename = f"{self.__len__}.pt"
        torch.save(stacked_activation_vectors, self._storage_path / filename)

        # Update the length
        if self.length_cache is not None:
            self.length_cache += len(stacked_activation_vectors)

    def append(self, item: ActivationStoreItem) -> Future | None:
        """Add a Single Item to the Store.

        Example:

        >>> store = DiskActivationStore(max_queue_size=1, empty_dir=True)
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
        self._queue.put(item)
        with self._queue_size_lock:
            self._queue_size.value += 1

        # Write to disk if needed
        if self._queue_size.value >= self._max_queue_size:
            return self._thread_pool.submit(self._write_to_disk)
        return None  # Keep mypy happy

    def extend(self, batch: ActivationStoreBatch) -> None:
        """Add a Batch to the Store.

        Args:
            batch: Batch of activation vectors to add to the store.
        """
        for item in batch:
            self.append(item)

    def close(self):
        """Close the Store (Make Sure All Activation Vectors are Written to Disk).

        This should be called after the last batch has been added to the store.
        """
        if not self._queue.empty():
            self._write_to_disk()

    @property
    def _all_filenames(self) -> list[Path]:
        """Return a List of All Activation Vector Filenames."""
        return [path for path in self._storage_path.glob("*.pt")]

    def empty(self):
        """Empty the Store.

        Warning:

        This will delete all .pt files in the top level of the storage directory.
        """
        for file in self._all_filenames:
            file.unlink()

        # Reset the length
        self.length_cache = 0

    def __getitem__(self, index: int) -> ActivationStoreItem:
        """Get Item Dunder Method.

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.
        """
        # Find the file containing the activation vector
        file_index = index // self._max_queue_size
        file = self._storage_path / f"{file_index}.pt"

        # Load the file and return the activation vector
        activation_vectors = torch.load(file)
        return activation_vectors[index % self._max_queue_size]

    def __len__(self) -> int:
        """Length Dunder Method.

        Example:

        >>> store = DiskActivationStore(max_queue_size=1, empty_dir=True)
        >>> print(len(store))
        0
        """
        # Calculate the length if not cached
        if self.length_cache is None:
            cache_size: int = 0
            for file in self._all_filenames:
                cache_size += len(torch.load(file))
            self.length_cache = cache_size

        return self.length_cache
