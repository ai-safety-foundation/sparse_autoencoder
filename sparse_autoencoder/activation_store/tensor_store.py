"""Tensor Activation Store."""
from jaxtyping import Float
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor

from sparse_autoencoder.activation_store.base_store import (
    ActivationStore,
    StoreFullError,
)
from sparse_autoencoder.tensor_types import Axis


class TensorActivationStore(ActivationStore):
    """Tensor Activation Store.

    Stores tensors in a (large) tensor of shape (item, neuron). Requires the number of activation
    vectors to be stored to be known in advance. Multiprocess safe.

    Extends the `torch.utils.data.Dataset` class to provide a list-based activation store, with
    additional :meth:`append` and :meth:`extend` methods (the latter of which is non-blocking).

    Examples:
    Create an empty activation dataset:

        >>> import torch
        >>> store = TensorActivationStore(max_items=1000, n_neurons=100, n_components=2)

    Add a single activation vector to the dataset (for a component):

        >>> store.append(torch.randn(100), component_idx=0)
        >>> store.append(torch.randn(100), component_idx=1)
        >>> len(store)
        1

    Add a [batch, neurons] activation tensor to the dataset:

        >>> store.empty()
        >>> batch = torch.randn(10, 100)
        >>> store.extend(batch, component_idx=0)
        >>> store.extend(batch, component_idx=1)
        >>> len(store)
        10

    Shuffle the dataset **before passing it to the DataLoader**:

        >>> store.shuffle() # Faster than using the DataLoader shuffle argument

    Use the dataloader to iterate over the dataset:

        >>> loader = torch.utils.data.DataLoader(store, shuffle=False, batch_size=2)
        >>> next_item = next(iter(loader))
        >>> next_item.shape
        torch.Size([2, 2, 100])
    """

    _data: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT, Axis.INPUT_OUTPUT_FEATURE)]
    """Underlying Tensor Data Store."""

    _items_stored: list[int]
    """Number of items stored."""

    max_items: int
    """Maximum Number of Items to Store."""

    _n_components: int
    """Number of components"""

    @property
    def n_components(self) -> int:
        """Number of components."""
        return self._n_components

    @property
    def current_activations_stored_per_component(self) -> list[int]:
        """Number of activations stored per component."""
        return self._items_stored

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        max_items: PositiveInt,
        n_neurons: PositiveInt,
        n_components: PositiveInt,
        device: torch.device | None = None,
    ) -> None:
        """Initialise the Tensor Activation Store.

        Args:
            max_items: Maximum number of items to store per component (individual activation
                vectors).
            n_neurons: Number of neurons in each activation vector.
            n_components: Number of components to store (i.e. number of source models).
            device: Device to store the activation vectors on.
        """
        self._n_components = n_components
        self._items_stored = [0] * n_components
        self._max_items = max_items
        self._data = torch.empty((max_items, n_components, n_neurons), device=device)

    def __len__(self) -> int:
        """Length Dunder Method.

        Returns the number of activation vectors per component in the dataset.

        Example:
            >>> import torch
            >>> store = TensorActivationStore(max_items=10_000_000, n_neurons=100, n_components=1)
            >>> store.append(torch.randn(100), component_idx=0)
            >>> store.append(torch.randn(100), component_idx=0)
            >>> len(store)
            2

        Returns:
            The number of activation vectors in the dataset.
        """
        # Min as this is the amount of activations that can be fetched by get_item
        return min(self.current_activations_stored_per_component)

    def __sizeof__(self) -> int:
        """Sizeof Dunder Method.

        Example:
            >>> import torch
            >>> store = TensorActivationStore(max_items=2, n_neurons=100, n_components=1)
            >>> store.__sizeof__() # Pre-allocated tensor of 2x100
            800

        Returns:
            The size of the underlying tensor in bytes.
        """
        return self._data.element_size() * self._data.nelement()

    def __getitem__(
        self, index: tuple[int, ...] | slice | int
    ) -> Float[Tensor, Axis.names(Axis.ANY)]:
        """Get Item Dunder Method.

        Examples:
            >>> import torch
            >>> store = TensorActivationStore(max_items=2, n_neurons=5, n_components=1)
            >>> store.append(torch.zeros(5), component_idx=0)
            >>> store.append(torch.ones(5), component_idx=0)
            >>> store[1, 0]
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
        >>> store = TensorActivationStore(max_items=10, n_neurons=1, n_components=1)
        >>> store.append(torch.tensor([0.]), component_idx=0)
        >>> store.append(torch.tensor([1.]), component_idx=0)
        >>> store.append(torch.tensor([2.]), component_idx=0)
        >>> store.shuffle()
        >>> [store[i, 0].item() for i in range(3)]
        [0.0, 2.0, 1.0]
        """
        # Generate a permutation of the indices for the active data
        perm = torch.randperm(len(self))

        # Use this permutation to shuffle the active data in-place
        self._data[: len(self)] = self._data[perm]

    def append(self, item: Float[Tensor, Axis.INPUT_OUTPUT_FEATURE], component_idx: int) -> None:
        """Add a single item to the store.

        Example:
        >>> import torch
        >>> store = TensorActivationStore(max_items=10, n_neurons=5, n_components=1)
        >>> store.append(torch.zeros(5), component_idx=0)
        >>> store.append(torch.ones(5), component_idx=0)
        >>> store[1, 0]
        tensor([1., 1., 1., 1., 1.])

        Args:
            item: The item to append to the dataset.
            component_idx: The component index to append the item to.

        Raises:
            IndexError: If there is no space remaining.
        """
        # Check we have space
        if self._items_stored[component_idx] + 1 > self._max_items:
            raise StoreFullError

        self._data[self._items_stored[component_idx], component_idx] = item.to(
            self._data.device,
        )
        self._items_stored[component_idx] += 1

    def extend(
        self,
        batch: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
        component_idx: int,
    ) -> None:
        """Add a batch to the store.

        Examples:
        >>> import torch
        >>> store = TensorActivationStore(max_items=10, n_neurons=5, n_components=1)
        >>> store.extend(torch.zeros(2, 5), component_idx=0)
        >>> len(store)
        2

        Args:
            batch: The batch to append to the dataset.
            component_idx: The component index to append the batch to.

        Raises:
            IndexError: If there is no space remaining.
        """
        # Check we have space
        n_activation_tensors: int = batch.shape[0]
        if self._items_stored[component_idx] + n_activation_tensors > self._max_items:
            raise StoreFullError

        self._data[
            self._items_stored[component_idx] : self._items_stored[component_idx]
            + n_activation_tensors,
            component_idx,
        ] = batch.to(self._data.device)
        self._items_stored[component_idx] += n_activation_tensors

    def empty(self) -> None:
        """Empty the store.

        Example:
        >>> import torch
        >>> store = TensorActivationStore(max_items=10, n_neurons=5, n_components=1)
        >>> store.extend(torch.zeros(2, 5), component_idx=0)
        >>> len(store)
        2
        >>> store.empty()
        >>> len(store)
        0
        """
        # We don't need to zero the data, just reset the number of items stored
        self._items_stored = [0 for _ in self._items_stored]
