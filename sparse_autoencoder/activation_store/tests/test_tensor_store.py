"""Tests for the TensorActivationStore."""
import pytest
import torch

from sparse_autoencoder.activation_store.base_store import StoreFullError
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore


def test_extended_items_all_returned_with_get() -> None:
    """Test that all items extended onto the store can be got back."""
    n_neurons: int = 128
    n_batches: int = 10
    batch_size: int = 16
    store = TensorActivationStore(max_items=int(n_batches * batch_size), n_neurons=n_neurons)

    batches = [torch.rand(batch_size, n_neurons) for _ in range(n_batches)]

    for batch in batches:
        store.extend(batch)

    assert len(store) == int(n_batches * batch_size)

    recovered_items = [store[i] for i in range(len(store))]
    all_batches_tensor = torch.cat(batches, dim=0)
    all_recovered_items_tensor = torch.stack(recovered_items, dim=0)
    assert torch.equal(all_batches_tensor, all_recovered_items_tensor)


def test_works_with_2_components() -> None:
    """Test that it works with 2 components."""
    n_neurons: int = 128
    n_batches: int = 10
    batch_size: int = 16
    store = TensorActivationStore(
        max_items=int(n_batches * batch_size), n_neurons=n_neurons, n_components=2
    )

    batches_component_0 = [torch.rand(batch_size, n_neurons) for _ in range(n_batches)]
    batches_component_1 = [torch.rand(batch_size, n_neurons) for _ in range(n_batches)]

    for batch_0, batch_1 in zip(batches_component_0, batches_component_1):
        store.extend(batch_0, component_idx=0)
        store.extend(batch_1, component_idx=1)

    assert len(store) == int(n_batches * batch_size)

    recovered_items = [store[i] for i in range(len(store))]

    all_batches_component_0_tensor = torch.cat(batches_component_0, dim=0)
    all_batches_component_1_tensor = torch.cat(batches_component_1, dim=0)
    all_batches_tensor = torch.stack(
        [all_batches_component_0_tensor, all_batches_component_1_tensor], dim=1
    )

    all_recovered_items_tensor = torch.stack(recovered_items, dim=0)

    assert torch.equal(all_batches_tensor, all_recovered_items_tensor)


def test_appending_more_than_max_items_raises() -> None:
    """Test that appending more than the max items raises an error."""
    n_neurons: int = 128
    store = TensorActivationStore(max_items=1, n_neurons=n_neurons)
    store.append(torch.rand(n_neurons))

    with pytest.raises(StoreFullError):
        store.append(torch.rand(n_neurons))


def test_extending_more_than_max_items_raises() -> None:
    """Test that extending more than the max items raises an error."""
    n_neurons: int = 128
    store = TensorActivationStore(max_items=6, n_neurons=n_neurons)
    store.extend(torch.rand(4, n_neurons))

    with pytest.raises(StoreFullError):
        store.extend(torch.rand(4, n_neurons))


def test_getting_out_of_range_raises() -> None:
    """Test that getting an out of range index raises an error."""
    n_neurons: int = 128
    store = TensorActivationStore(max_items=1, n_neurons=n_neurons)
    store.append(torch.rand(n_neurons))

    with pytest.raises(IndexError):
        store[1]
