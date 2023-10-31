"""List Activation Store."""
import random
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Manager
from multiprocessing.managers import ListProxy

import torch

from sparse_autoencoder.activations.ActivationStore import (
    ActivationStore,
    ActivationStoreBatch,
    ActivationStoreItem,
)


class ListActivationStore(ActivationStore):
    """List Activation Store.

    Stores pointers to activation vectors in a list (in-memory). Multiprocess safe if the
    `multiprocessing_enabled` argument is set to `True`.

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
        >>> future = store.extend(batch)
        >>> future.result() # Wait for the write to complete
        >>> len(store)
        11

    Shuffle the dataset **before passing it to the DataLoader**:

        >>> store.shuffle() # Faster than using the DataLoader shuffle argument
        >>> loader = torch.utils.data.DataLoader(store, shuffle=False, batch_size=2)

    Use the dataloader to iterate over the dataset:

        >>> next_item = next(iter(loader))
        >>> next_item.shape
        torch.Size([2, 100])

    Args:
        data: Data to initialize the dataset with.
        device: Device to store the activation vectors on.
        dtype: Data type to store the activation vectors as.
        multiprocessing_enabled: Support reading/writing to the dataset with multiple
            GPU workers. This creates significant overhead, so you should only enable it if you
            have multiple GPUs (and experiment with enabling/disabling it).
        num_workers: Number of CPU workers to use for non-blocking writes to the dataset (so that
            the model can keep running whilst it writes the previous activations to memory). This
            should be less than the number of CPU cores available. You don't need multiple GPUs to
            take advantage of this feature.
    """

    _data: list[ActivationStoreItem] | ListProxy
    """Underlying List Data Store."""

    _device: torch.device
    """Device to Store the Activation Vectors On."""

    _dtype: torch.dtype
    """Data Type to Store the Activation Vectors As."""

    _thread_pool: ThreadPoolExecutor
    """Threadpool for non-blocking writes to the dataset."""

    def __init__(
        self,
        data: list[ActivationStoreItem] | None = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
        multiprocessing_enabled=False,
        num_workers: int = 6,
    ) -> None:
        # Default to empty
        if data is None:
            data = []

        # If multiprocessing is enabled, use a multiprocessing manager to create a shared list
        # between processes. Otherwise, just use a normal list.
        if multiprocessing_enabled:
            manager = Manager()
            self._data = manager.list(data)
        else:
            self._data = data

        # Device and dtype for storing the activation vectors
        self._device = device
        self._dtype = dtype

        # Create a threadpool for non-blocking writes to the dataset
        self._thread_pool = ThreadPoolExecutor(num_workers)

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
        """
        return len(self._data)

    def __sizeof__(self) -> int:
        """Sizeof Dunder Method.

        Returns the size of the dataset in bytes.
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

    def __getitem__(self, index: int) -> ActivationStoreItem:
        """Get Item Dunder Method.

        Example:

        >>> import torch
        >>> store = ListActivationStore()
        >>> store.append(torch.zeros(5))
        >>> store.append(torch.ones(5))
        >>> store[1]
        tensor([1., 1., 1., 1., 1.], dtype=torch.float16)

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.
        """
        return self._data[index]

    def shuffle(self):
        """Shuffle the Data In-Place.

        This is much faster than using the shuffle argument on `torch.utils.data.DataLoader`.

        Example:

        >>> import torch
        >>> import random
        >>> random.seed(42)
        >>> store = ListActivationStore()
        >>> store.append(torch.tensor([1.]))
        >>> store.append(torch.tensor([2.]))
        >>> store.append(torch.tensor([3.]))
        >>> store.shuffle()
        >>> [store[i].item() for i in range(len(store))]
        [2.0, 1.0, 3.0]

        """
        self.wait_for_writes_to_complete()
        random.shuffle(self._data)

    def append(self, item: ActivationStoreItem) -> None:
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
        """
        self._data.append(item.to(self._device, self._dtype))

    def _extend(self, batch: ActivationStoreBatch) -> None:
        """Extend threadpool method.

        To be called by :meth:`extend`.

        Args:
            items: A list of items to add to the dataset.
        """
        # Unstack to a list of tensors
        items: list[ActivationStoreItem] = batch.to(self._device, self._dtype).unbind(0)

        self._data.extend(items)

    def extend(self, batch: ActivationStoreBatch) -> Future:
        """Extend the dataset with multiple items (non-blocking).

        Example:

            >>> import torch
            >>> store = ListActivationStore()
            >>> batch = torch.randn(10, 100)
            >>> future = store.extend(batch)
            >>> future.result() # Wait for the write to complete
            >>> len(store)
            10

        Args:
            items: A list of items to add to the dataset.
        """
        return self._thread_pool.submit(self._extend, batch)

    def wait_for_writes_to_complete(self):
        """Wait for Writes to Complete

        Wait for any non-blocking writes (e.g. calls to :meth:`append`) to complete.

        Example:

        >>> import torch
        >>> store = ListActivationStore()
        >>> _future = store.extend(torch.randn(3, 100))
        >>> len(store) # The writes haven't completed yet
        0

        >>> store.wait_for_writes_to_complete()
        >>> len(store)
        3
        """
        # Submit a dummy task to the thread pool
        sentinel = object()
        future = self._thread_pool.submit(lambda: sentinel)
        future.result()

    def __del__(self):
        """Delete Dunder Method."""
        # Shutdown the thread pool (don't wait as we won't be able to use the resulting dataset)
        self._thread_pool.shutdown(wait=False, cancel_futures=True)
