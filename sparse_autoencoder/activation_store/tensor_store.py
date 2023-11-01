"""Tensor Activation Store."""
import torch
from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.activation_store.base_store import (
    ActivationStore,
    ActivationStoreBatch,
    ActivationStoreItem,
    StoreFullError,
)
from sparse_autoencoder.activation_store.utils.extend_resize import (
    resize_to_single_item_dimension,
)

TensorActivationStoreData = Float[Tensor, "item neuron"]
"""Tensor Activation Store Dataset Item Type."""


class TensorActivationStore(ActivationStore):
    """Tensor Activation Store.

    Stores tensors in a (large) tensor of shape (item, neuron). Requires the number of activation
    vectors to be stored to be known in advance. Multiprocess safe.

    Extends the `torch.utils.data.Dataset` class to provide a list-based activation store, with
    additional :meth:`append` and :meth:`extend` methods (the latter of which is non-blocking).

    Examples:

    Create an empty activation dataset:

        >>> import torch
        >>> store = TensorActivationStore(max_items=1000, num_neurons=100)

    Add a single activation vector to the dataset:

        >>> store.append(torch.randn(100))
        >>> len(store)
        1

    Add a [batch, pos, neurons] activation tensor to the dataset:

        >>> store.empty()
        >>> batch = torch.randn(10, 10, 100)
        >>> store.extend(batch)
        >>> len(store)
        100

    Shuffle the dataset **before passing it to the DataLoader**:

        >>> store.shuffle() # Faster than using the DataLoader shuffle argument
        >>> loader = torch.utils.data.DataLoader(store, shuffle=False, batch_size=2)

    Use the dataloader to iterate over the dataset:

        >>> next_item = next(iter(loader))
        >>> next_item.shape
        torch.Size([2, 100])

    Args:
        max_items: Maximum number of items to store (individual activation vectors)
        num_neurons: Number of neurons in each activation vector.
        device: Device to store the activation vectors on.
        dtype: Data type to store the activation vectors as.
    """

    _data: TensorActivationStoreData
    """Underlying Tensor Data Store."""

    items_stored: int = 0
    """Number of items stored."""

    max_items: int
    """Maximum Number of Items to Store."""

    def __init__(
        self,
        max_items: int,
        num_neurons: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
    ) -> None:
        # Initialise the datastore
        self._data = torch.empty((max_items, num_neurons), device=device, dtype=dtype)
        self._max_items = max_items

    def __len__(self) -> int:
        """Length Dunder Method.

        Returns the number of activation vectors in the dataset.

        Example:

        >>> import torch
        >>> store = TensorActivationStore(max_items=10_000_000, num_neurons=100)
        >>> store.append(torch.randn(100))
        >>> store.append(torch.randn(100))
        >>> len(store)
        2
        """
        return self.items_stored

    def __sizeof__(self) -> int:
        """Sizeof Dunder Method.

        Returns the size of the underlying tensor in bytes.

        Example:

        >>> import torch
        >>> store = TensorActivationStore(max_items=2, num_neurons=100)
        >>> store.__sizeof__() # Pre-allocated tensor of 2x100
        400
        """
        return self._data.element_size() * self._data.nelement()

    def __getitem__(self, index: int) -> ActivationStoreItem:
        """Get Item Dunder Method.

        Example:

        >>> import torch
        >>> store = TensorActivationStore(max_items=2, num_neurons=5)
        >>> store.append(torch.zeros(5))
        >>> store.append(torch.ones(5))
        >>> store[1]
        tensor([1., 1., 1., 1., 1.], dtype=torch.float16)

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        # Check in range
        if index >= self.items_stored:
            raise IndexError(
                f"Index {index} out of range (only {self.items_stored} items stored)"
            )

        return self._data[index]

    def shuffle(self):
        """Shuffle the Data In-Place.

        This is much faster than using the shuffle argument on `torch.utils.data.DataLoader`.

        Example:

        >>> import torch
        >>> _seed = torch.manual_seed(42)
        >>> store = TensorActivationStore(max_items=10, num_neurons=1)
        >>> store.append(torch.tensor([0.]))
        >>> store.append(torch.tensor([1.]))
        >>> store.append(torch.tensor([2.]))
        >>> store.shuffle()
        >>> [store[i].item() for i in range(3)]
        [0.0, 2.0, 1.0]
        """
        # Generate a permutation of the indices for the active data
        perm = torch.randperm(self.items_stored)

        # Use this permutation to shuffle the active data in-place
        self._data[: self.items_stored] = self._data[perm]

    def append(self, item: ActivationStoreItem) -> None:
        """Add a single item to the store.

        Example:

        >>> import torch
        >>> store = TensorActivationStore(max_items=10, num_neurons=5)
        >>> store.append(torch.zeros(5))
        >>> store.append(torch.ones(5))
        >>> store[1]
        tensor([1., 1., 1., 1., 1.], dtype=torch.float16)

        Args:
            item: The item to append to the dataset.

        Raises:
            IndexError: If there is no space remaining.
        """
        # Check we have space
        if self.items_stored + 1 > self._max_items:
            raise StoreFullError()

        self._data[self.items_stored] = item.to(
            self._data.device, dtype=self._data.dtype
        )
        self.items_stored += 1

    def extend(self, batch: ActivationStoreBatch) -> None:
        """Add a batch to the store.

        Examples:

        >>> import torch
        >>> store = TensorActivationStore(max_items=10, num_neurons=5)
        >>> store.extend(torch.zeros(2, 5))
        >>> store.items_stored
        2

        >>> store = TensorActivationStore(max_items=10, num_neurons=5)
        >>> store.extend(torch.zeros(3, 3, 5))
        >>> store.items_stored
        9

        Args:
            batch: The batch to append to the dataset.

        Raises:
            IndexError: If there is no space remaining.
        """
        reshaped: Float[Tensor, "subset_item neuron"] = resize_to_single_item_dimension(
            batch
        )

        # Check we have space
        if self.items_stored + reshaped.shape[0] > self._max_items:
            raise StoreFullError()

        n_items = reshaped.shape[0]
        self._data[self.items_stored : self.items_stored + n_items] = reshaped.to(
            self._data.device, dtype=self._data.dtype
        )
        self.items_stored += n_items

    def empty(self) -> None:
        """Empty the store.

        Example:

        >>> import torch
        >>> store = TensorActivationStore(max_items=10, num_neurons=5)
        >>> store.extend(torch.zeros(2, 5))
        >>> store.items_stored
        2
        >>> store.empty()
        >>> store.items_stored
        0
        """
        # We don't need to zero the data, just reset the number of items stored
        self.items_stored = 0
