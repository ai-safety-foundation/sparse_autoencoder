"""List Activation Store."""
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from multiprocessing.managers import ListProxy
import random
import time

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


class ListActivationStore(ActivationStore):
    """List Activation Store.

    Stores pointers to activation vectors in a list (in-memory). This is primarily of use for quick
    experiments where you don't want to calculate how much memory you need in advance.

    Multiprocess safe if the `multiprocessing_enabled` argument is set to `True`. This works in two
    ways:

    1. The list of activation vectors is stored in a multiprocessing manager, which allows multiple
        processes (typically multiple GPUs) to read/write to the list.
    2. The `extend` method is non-blocking, and uses a threadpool to write to the list in the
        background, which allows the main process to continue working even if there is just one GPU.

    Extends the `torch.utils.data.Dataset` class to provide a list-based activation store, with
    additional :meth:`append` and :meth:`extend` methods (the latter of which is non-blocking).

    Note that the built-in :meth:`shuffle` method is much faster than using the `shuffle` argument
    on `torch.utils.data.DataLoader`. You should therefore call this method before passing the
    dataset to the loader and then set the DataLoader `shuffle` argument to `False`.

    Examples:
    Create an empty activation dataset:

        >>> import torch
        >>> store = ListActivationStore()

    Add a single activation vector to the dataset (this is blocking):

        >>> store.append(torch.randn(100))
        >>> len(store)
        1

    Add a batch of activation vectors to the dataset (non-blocking):

        >>> batch = torch.randn(10, 100)
        >>> store.extend(batch)
        >>> len(store)
        11

    Shuffle the dataset **before passing it to the DataLoader**:

        >>> store.shuffle() # Faster than using the DataLoader shuffle argument
        >>> loader = torch.utils.data.DataLoader(store, shuffle=False, batch_size=2)

    Use the dataloader to iterate over the dataset:

        >>> next_item = next(iter(loader))
        >>> next_item.shape
        torch.Size([2, 100])
    """

    _data: list[InputOutputActivationVector] | ListProxy
    """Underlying List Data Store."""

    _device: torch.device | None
    """Device to Store the Activation Vectors On."""

    _pool: ProcessPoolExecutor | None = None
    """Multiprocessing Pool."""

    _pool_exceptions: ListProxy | list[Exception]
    """Pool Exceptions.

    Used to keep track of exceptions.
    """

    _pool_futures: list[Future]
    """Pool Futures.

    Used to keep track of processes running in the pool.
    """

    def __init__(
        self,
        data: list[InputOutputActivationVector] | None = None,
        device: torch.device | None = None,
        max_workers: int | None = None,
        *,
        multiprocessing_enabled: bool = False,
    ) -> None:
        """Initialize the List Activation Store.

        Args:
            data: Data to initialize the dataset with.
            device: Device to store the activation vectors on.
            max_workers: Max CPU workers if multiprocessing is enabled, for writing to the list.
                Default is the number of cores you have.
            multiprocessing_enabled: Support reading/writing to the dataset with multiple GPU
                workers. This creates significant overhead, so you should only enable it if you have
                multiple GPUs (and experiment with enabling/disabling it).
        """
        # Default to empty
        if data is None:
            data = []

        # If multiprocessing is enabled, use a multiprocessing manager to create a shared list
        # between processes. Otherwise, just use a normal list.
        if multiprocessing_enabled:
            self._pool = ProcessPoolExecutor(max_workers=max_workers)
            manager = Manager()
            self._data = manager.list(data)
            self._data.extend(data)
            self._pool_exceptions = manager.list()
        else:
            self._data = data
            self._pool_exceptions = []

        self._pool_futures = []

        # Device for storing the activation vectors
        self._device = device

    def __len__(self) -> int:
        """Length Dunder Method.

        Returns the number of activation vectors in the dataset.

        Example:
            >>> import torch
            >>> store = ListActivationStore()
            >>> store.append(torch.randn(100))
            >>> store.append(torch.randn(100))
            >>> len(store)
            2

        Returns:
            The number of activation vectors in the dataset.
        """
        return len(self._data)

    def __sizeof__(self) -> int:
        """Sizeof Dunder Method.

        Returns:
            The size of the dataset in bytes.
        """
        # The list of tensors is really a list of pointers to tensors, so we need to account for
        # this as well as the size of the tensors themselves.
        list_of_pointers_size = self._data.__sizeof__()

        # Handle 0 items
        if len(self._data) == 0:
            return list_of_pointers_size

        # Otherwise, get the size of the first tensor
        first_tensor = self._data[0]
        first_tensor_size = first_tensor.element_size() * first_tensor.nelement()
        num_tensors = len(self._data)
        total_tensors_size = first_tensor_size * num_tensors

        return total_tensors_size + list_of_pointers_size

    def __getitem__(self, index: int) -> InputOutputActivationVector:
        """Get Item Dunder Method.

        Example:
        >>> import torch
        >>> store = ListActivationStore()
        >>> store.append(torch.zeros(5))
        >>> store.append(torch.ones(5))
        >>> store[1]
        tensor([1., 1., 1., 1., 1.])

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.
        """
        return self._data[index]

    def shuffle(self) -> None:
        """Shuffle the Data In-Place.

        This is much faster than using the shuffle argument on `torch.utils.data.DataLoader`.

        Example:
        >>> import torch
        >>> _seed = torch.manual_seed(42)
        >>> store = ListActivationStore()
        >>> store.append(torch.tensor([1.]))
        >>> store.append(torch.tensor([2.]))
        >>> store.append(torch.tensor([3.]))
        >>> store.shuffle()
        >>> len(store)
        3

        """
        self.wait_for_writes_to_complete()
        random.shuffle(self._data)

    def append(self, item: InputOutputActivationVector) -> Future | None:
        """Append a single item to the dataset.

        Note **append is blocking**. For better performance use extend instead with batches.

        Example:
        >>> import torch
        >>> store = ListActivationStore()
        >>> store.append(torch.randn(100))
        >>> store.append(torch.randn(100))
        >>> len(store)
        2

        Args:
            item: The item to append to the dataset.

        Returns:
            Future that completes when the activation vector has queued to be written to disk, and
            if needed, written to disk.
        """
        self._data.append(item.to(self._device))

    def _extend(self, batch: SourceModelActivations) -> None:
        """Extend threadpool method.

        To be called by :meth:`extend`.

        Args:
            batch: A batch of items to add to the dataset.
        """
        try:
            # Unstack to a list of tensors
            items: list[InputOutputActivationVector] = resize_to_list_vectors(batch)

            self._data.extend(items)
        except Exception as e:  # noqa: BLE001
            self._pool_exceptions.append(e)

    def extend(self, batch: SourceModelActivations) -> Future | None:
        """Extend the dataset with multiple items (non-blocking).

        Example:
            >>> import torch
            >>> store = ListActivationStore()
            >>> batch = torch.randn(10, 100)
            >>> async_result = store.extend(batch)
            >>> len(store)
            10

        Args:
            batch: A batch of items to add to the dataset.

        Returns:
            Future that completes when the activation vectors have queued to be written to disk, and
            if needed, written to disk.
        """
        # Schedule _extend to run in a separate process
        if self._pool:
            future = self._pool.submit(self._extend, batch)
            self._pool_futures.append(future)

        # Fallback to synchronous execution if not multiprocessing
        self._extend(batch)

    def wait_for_writes_to_complete(self) -> None:
        """Wait for Writes to Complete.

        Wait for any non-blocking writes (e.g. calls to :meth:`append`) to complete.

        Example:
            >>> import torch
            >>> store = ListActivationStore(multiprocessing_enabled=True)
            >>> store.extend(torch.randn(3, 100))
            >>> store.wait_for_writes_to_complete()
            >>> len(store)
            3

        Raises:
            RuntimeError: If any exceptions occurred in the background workers.
        """
        # Restart the pool
        if self._pool:
            for _future in as_completed(self._pool_futures):
                pass
            self._pool_futures.clear()

        time.sleep(1)

        if self._pool_exceptions:
            exceptions_report = "\n".join([str(e) for e in self._pool_exceptions])
            msg = f"Exceptions occurred in background workers:\n{exceptions_report}"
            raise RuntimeError(msg)

    def empty(self) -> None:
        """Empty the dataset.

        Example:
        >>> import torch
        >>> store = ListActivationStore()
        >>> store.append(torch.randn(100))
        >>> store.append(torch.randn(100))
        >>> len(store)
        2

        >>> store.empty()
        >>> len(store)
        0
        """
        self.wait_for_writes_to_complete()

        # Clearing a list like this works for both standard and multiprocessing lists
        self._data[:] = []

    def __del__(self) -> None:
        """Delete Dunder Method."""
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
