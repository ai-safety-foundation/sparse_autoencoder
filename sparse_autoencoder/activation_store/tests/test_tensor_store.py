"""Tests for the TensorActivationStore."""
import pytest
import torch

from sparse_autoencoder.activation_store.base_store import StoreFullError
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore


def test_works_with_2_components() -> None:
    """Test that it works with 2 components."""
    n_neurons: int = 128
    n_batches: int = 10
    batch_size: int = 16
    n_components: int = 2

    store = TensorActivationStore(
        max_items=int(n_batches * batch_size), n_neurons=n_neurons, n_components=n_components
    )

    # Fill the store
    for component_idx in range(n_components):
        for _ in range(n_batches):
            store.extend(torch.rand(batch_size, n_neurons), component_idx=component_idx)

    # Check the size
    assert len(store) == int(n_batches * batch_size)


def test_appending_more_than_max_items_raises() -> None:
    """Test that appending more than the max items raises an error."""
    n_neurons: int = 128
    store = TensorActivationStore(max_items=1, n_neurons=n_neurons, n_components=1)
    store.append(torch.rand(n_neurons), component_idx=0)

    with pytest.raises(StoreFullError):
        store.append(torch.rand(n_neurons), component_idx=0)


def test_extending_more_than_max_items_raises() -> None:
    """Test that extending more than the max items raises an error."""
    n_neurons: int = 128
    store = TensorActivationStore(max_items=6, n_neurons=n_neurons, n_components=1)
    store.extend(torch.rand(4, n_neurons), component_idx=0)

    with pytest.raises(StoreFullError):
        store.extend(torch.rand(4, n_neurons), component_idx=0)


def test_getting_out_of_range_raises() -> None:
    """Test that getting an out of range index raises an error."""
    n_neurons: int = 128
    store = TensorActivationStore(max_items=1, n_neurons=n_neurons, n_components=1)
    store.append(torch.rand(n_neurons), component_idx=0)

    with pytest.raises(IndexError):
        store[1, 0]
