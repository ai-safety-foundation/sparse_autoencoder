"""Tensor Activation Store."""
import torch

from sparse_autoencoder.activation_store.base_store import (
    ActivationStore,
    StoreFullError,
)
from sparse_autoencoder.activation_store.utils.extend_resize import (
    resize_to_single_item_dimension,
)
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    InputOutputActivationVector,
    SourceModelActivations,
    StoreActivations,
)


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
    """

    _data: StoreActivations
    """Underlying Tensor Data Store."""

    items_stored: int = 0
    """Number of items stored."""

    max_items: int
    """Maximum Number of Items to Store."""

    def __init__(
        self,
        max_items: int,
        num_neurons: int,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the Tensor Activation Store.

        Args:
            max_items: Maximum number of items to store (individual activation vectors)
            num_neurons: Number of neurons in each activation vector.
            device: Device to store the activation vectors on.
        """
        self._data = torch.empty((max_items, num_neurons), device=device)
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

        Returns:
            The number of activation vectors in the dataset.
        """
        return self.items_stored

    def __sizeof__(self) -> int:
        """Sizeof Dunder Method.

        Example:
            >>> import torch
            >>> store = TensorActivationStore(max_items=2, num_neurons=100)
            >>> store.__sizeof__() # Pre-allocated tensor of 2x100
            800

        Returns:
            The size of the underlying tensor in bytes.
        """
        return self._data.element_size() * self._data.nelement()

    def __getitem__(self, index: int) -> InputOutputActivationVector:
        """Get Item Dunder Method.

        Example:
        >>> import torch
        >>> store = TensorActivationStore(max_items=2, num_neurons=5)
        >>> store.append(torch.zeros(5))
        >>> store.append(torch.ones(5))
        >>> store[1]
        tensor([1., 1., 1., 1., 1.])

        Args:
            index: The index of the tensor to fetch.

        Returns:
            The activation store item at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        # Check in range
        if index >= self.items_stored:
            msg = f"Index {index} out of range (only {self.items_stored} items stored)"
            raise IndexError(msg)

        return self._data[index]

    def shuffle(self) -> None:
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

    def append(self, item: InputOutputActivationVector) -> None:
        """Add a single item to the store.

        Example:
        >>> import torch
        >>> store = TensorActivationStore(max_items=10, num_neurons=5)
        >>> store.append(torch.zeros(5))
        >>> store.append(torch.ones(5))
        >>> store[1]
        tensor([1., 1., 1., 1., 1.])

        Args:
            item: The item to append to the dataset.

        Raises:
            IndexError: If there is no space remaining.
        """
        # Check we have space
        if self.items_stored + 1 > self._max_items:
            raise StoreFullError

        self._data[self.items_stored] = item.to(
            self._data.device,
        )
        self.items_stored += 1

    def extend(self, batch: SourceModelActivations) -> None:
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
        reshaped: InputOutputActivationBatch = resize_to_single_item_dimension(
            batch,
        )

        # Check we have space
        num_activation_tensors: int = reshaped.shape[0]
        if self.items_stored + num_activation_tensors > self._max_items:
            if reshaped.shape[0] > self._max_items:
                msg = f"Single batch of {num_activation_tensors} activations is larger than the \
                    total maximum in the store of {self._max_items}."
                raise ValueError(msg)

            raise StoreFullError

        self._data[self.items_stored : self.items_stored + num_activation_tensors] = reshaped.to(
            self._data.device
        )
        self.items_stored += num_activation_tensors

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
