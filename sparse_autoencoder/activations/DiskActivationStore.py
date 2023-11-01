"""Disk Activation Store."""
import tempfile
from multiprocessing import Queue
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

    This class will delete all .pt files in the top level of the storage directory when initialized,
    unless the `empty_dir` argument is set to `False`.

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
    """

    _storage_path: Path
    """Path to the Directory where the Activation Vectors are Stored."""

    _queue: Queue
    """Queue to Buffer Activation Vectors in Memory.
    
    Activation vectors are buffered in memory until the queue is full, at which point they are
    written to disk.
    """

    __len__: int
    """Number of Activation Vectors in the Store."""

    _max_queue_size: int
    """Maximum Number of Activation Vectors to Buffer in Memory.
    
    Note this is approximately respected, as multiprocessing doesn't have strict guarantees.
    """

    def __init__(
        self,
        storage_path: Path = DEFAULT_DISK_ACTIVATION_STORE_PATH,
        empty_dir: bool = True,
        max_queue_size: int = 10_000,
    ):
        super().__init__()

        # Setup the storage directory
        self._storage_path = storage_path
        self._storage_path.mkdir(parents=True, exist_ok=True)
        if empty_dir:
            self.empty()

        # Setup the Queue
        # Note we don't explicitly set the maxsize of the queue, as we want to allow it to grow
        # beyond the max_queue_size if the workers are slow to write to disk (as this will block
        # the model from running otherwise).
        self._queue = Queue()
        self._max_queue_size = max_queue_size

    def _write_to_disk(self) -> None:
        """Write the contents of the queue to disk."""
        all_queue_items = [self._queue.get() for _ in range(self._queue.qsize())]
        stacked_activation_vectors = torch.stack(all_queue_items)
        filename = f"{self.__len__}.pt"
        torch.save(stacked_activation_vectors, self._storage_path / filename)

        # Update the length
        self.__len__ += len(stacked_activation_vectors)

    def append(self, item: ActivationStoreItem) -> None:
        """Add a Single Item to the Store.

        Args:
            item: Activation vector to add to the store.

        Example:

        >>> store = DiskActivationStore(max_queue_size=1)
        >>> store.append(torch.randn(100))
        >>> print(len(store))
        1
        """
        self._queue.put(item)
        if self._queue.qsize() >= self._max_queue_size:
            self._write_to_disk()

    def extend(self, batch: ActivationStoreBatch):
        """Add a Batch to the Store.

        Args:
            batch: Batch of activation vectors to add to the store.
        """
        for item in batch:
            # TODO: Make this non-blocking
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
        self.__len__ = 0

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
