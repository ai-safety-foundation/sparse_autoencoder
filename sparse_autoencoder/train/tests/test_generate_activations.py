"""Test Generate Activations."""

import pytest
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.list_store import ListActivationStore
from sparse_autoencoder.source_data.random_int import RandomIntDummyDataset
from sparse_autoencoder.train.generate_activations import generate_activations


@pytest.mark.skip(reason="We're changing the approach")
def test_activations_generated() -> None:
    """Check that activations are added to the store."""
    store = ListActivationStore()
    model = HookedTransformer.from_pretrained("tiny-stories-1M")

    batch_size = 2
    dataset = RandomIntDummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size)  # type: ignore

    num_items = 2

    generate_activations(
        model=model,
        layer=1,
        cache_name="blocks.1.mlp.hook_post",
        store=store,
        source_data=iter(dataloader),
        num_items=num_items,
        context_size=dataset.context_size,
        batch_size=2,
    )

    assert len(store) >= num_items
