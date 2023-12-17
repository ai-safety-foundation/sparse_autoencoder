"""Tests for the DiskActivationStore."""
import pytest
import torch

from sparse_autoencoder.activation_store.disk_store import DiskActivationStore


@pytest.skip("Disk check doesn't work on CD.")
def test_extended_items_all_returned_with_get() -> None:
    """Test that all items extended onto the store can be got back."""
    num_neurons: int = 128
    num_batches: int = 10
    batch_size: int = 16
    store = DiskActivationStore(
        max_cache_size=int(num_batches * batch_size), num_neurons=num_neurons
    )

    batches = [torch.rand(batch_size, num_neurons) for _ in range(num_batches)]

    for batch in batches:
        store.extend(batch)
    store.finalise()

    assert len(store) == int(num_batches * batch_size)

    recovered_items = [store[i] for i in range(len(store))]
    all_batches_tensor = torch.cat(batches, dim=0)
    all_recovered_items_tensor = torch.stack(recovered_items, dim=0)

    assert torch.equal(all_batches_tensor, all_recovered_items_tensor)
