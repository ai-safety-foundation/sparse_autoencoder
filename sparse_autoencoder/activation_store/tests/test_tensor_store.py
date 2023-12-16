"""Tests for the TensorActivationStore."""
import pytest
import torch

from sparse_autoencoder.activation_store.base_store import StoreFullError
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore


def test_extended_items_all_returned_with_get() -> None:
    """Test that all items extended onto the store can be got back."""
    num_neurons: int = 128
    num_batches: int = 10
    batch_size: int = 16
    store = TensorActivationStore(max_items=int(num_batches * batch_size), num_neurons=num_neurons)

    batches = [torch.rand(batch_size, num_neurons) for _ in range(num_batches)]

    for batch in batches:
        store.extend(batch)

    assert len(store) == int(num_batches * batch_size)

    recovered_items = [store[i] for i in range(len(store))]
    all_batches_tensor = torch.cat(batches, dim=0)
    all_recovered_items_tensor = torch.stack(recovered_items, dim=0)
    assert torch.equal(all_batches_tensor, all_recovered_items_tensor)


def test_appending_more_than_max_items_raises() -> None:
    """Test that appending more than the max items raises an error."""
    num_neurons: int = 128
    store = TensorActivationStore(max_items=1, num_neurons=num_neurons)
    store.append(torch.rand(num_neurons))

    with pytest.raises(StoreFullError):
        store.append(torch.rand(num_neurons))


def test_extending_more_than_max_items_raises() -> None:
    """Test that extending more than the max items raises an error."""
    num_neurons: int = 128
    store = TensorActivationStore(max_items=6, num_neurons=num_neurons)
    store.extend(torch.rand(4, num_neurons))

    with pytest.raises(StoreFullError):
        store.extend(torch.rand(4, num_neurons))


def test_getting_out_of_range_raises() -> None:
    """Test that getting an out of range index raises an error."""
    num_neurons: int = 128
    store = TensorActivationStore(max_items=1, num_neurons=num_neurons)
    store.append(torch.rand(num_neurons))

    with pytest.raises(IndexError):
        store[1]
