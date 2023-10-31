"""Activation Store Base Class."""
from abc import abstractmethod

from jaxtyping import Float
from torch import Tensor
from torch.utils.data import Dataset

ActivationStoreItem = Float[Tensor, "neuron"]
"""Activation Store Dataset Item Type."""

ActivationStoreBatch = Float[Tensor, "batch neuron"]
"""Activation Store Dataset Batch Type."""


class ActivationStore(Dataset):
    """Activation Store Abstract Class.

    Extends the `torch.utils.data.Dataset` class to provide an activation store, with additional
    :meth:`append` and :meth:`extend` methods (the latter of which should be non-blocking).

    Can be used with a `torch.utils.data.DataLoader` to iterate over the dataset.
    """

    @abstractmethod
    def append(self, item: ActivationStoreItem) -> None:
        """Add a single item to the store"""
        raise NotImplementedError

    @abstractmethod
    def extend(self, batch: ActivationStoreBatch):
        """Add a batch to the store"""
        raise NotImplementedError
