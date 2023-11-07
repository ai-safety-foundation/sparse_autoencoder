"""Activation Store Base Class."""
from abc import ABC, abstractmethod
from concurrent.futures import Future
from collections.abc import Callable

from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset


ActivationStoreItem = Float[Tensor, "neuron"]
"""Activation Store Dataset Item Type.

A single vector containing activations. For example this could be the activations from a specific
MLP layer, for a specific position and batch item.
"""

ActivationStoreBatch = Float[Tensor, "item neuron"]
"""Activation Store Dataset Batch Type.

A batch of activation vectors. For example this could be the activations from a specific MLP layer,
for a specific position and batch item.
"""

UnshapedActivationBatch = Float[Tensor, "*any neuron"]
"""Unshaped Activation Batch Type.

A batch of activation vectors, with arbitrary numbers of dimensions. For example this could be the
activations from a specific attention layer, for a specific position and batch item, with an
additional head_idx dimension.
"""

ReshapeMethod = Callable[[UnshapedActivationBatch], ActivationStoreBatch]
"""Reshape Method Type.

This is a function that takes a batch of activations and returns the activations reshaped into a
batched format. For example, this could be a function that takes a [batch, pos, neurons] tensor
and returns a [batch * pos, neurons] tensor, or a function that takes a [batch, pos, head_idx,
neurons] tensor and returns a [batch * pos * head_idx, neurons] tensor.
"""

class ActivationStore(Dataset[ActivationStoreItem], ABC):
    """Activation Store Abstract Class.

    Extends the `torch.utils.data.Dataset` class to provide an activation store, with additional
    :meth:`append` and :meth:`extend` methods (the latter of which should typically be
    non-blocking). The resulting activation store can be used with a `torch.utils.data.DataLoader`
    to iterate over the dataset.

    Extend this class if you want to create a new activation store (noting you also need to create
    `__getitem__` and `__len__` methods from the underlying `torch.utils.data.Dataset` class).

    Example:
    >>> import torch
    >>> class MyActivationStore(ActivationStore):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self._data = [] # In this example, we just store in a list
    ...
    ...     def append(self, item: ActivationStoreItem) -> None:
    ...         self._data.append(item)
    ...
    ...     def extend(self, batch: ActivationStoreBatch):
    ...         self._data.extend(batch)
    ...
    ...     def empty(self):
    ...         self._data = []
    ...
    ...     def __getitem__(self, index: int) -> ActivationStoreItem:
    ...         return self._data[index]
    ...
    ...     def __len__(self) -> int:
    ...         return len(self._data)
    ...
    >>> store = MyActivationStore()
    >>> store.append(torch.randn(100))
    >>> print(len(store))
    1
    """

    @abstractmethod
    def append(self, item: ActivationStoreItem) -> Future | None:
        """Add a Single Item to the Store."""
        raise NotImplementedError

    @abstractmethod
    def extend(self, batch: ActivationStoreBatch) -> Future | None:
        """Add a Batch to the Store."""
        raise NotImplementedError

    @abstractmethod
    def empty(self) -> None:
        """Empty the Store."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Get the Length of the Store."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> ActivationStoreItem:
        """Get an Item from the Store."""
        raise NotImplementedError

    def shuffle(self) -> None:
        """Optional shuffle method."""


class StoreFullError(IndexError):
    """Exception raised when the activation store is full."""

    def __init__(self, message: str = "Activation store is full"):
        """Initialise the exception.

        Args:
            message: Override the default message.
        """
        super().__init__(message)
