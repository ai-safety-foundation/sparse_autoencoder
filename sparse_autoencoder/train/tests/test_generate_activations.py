"""Test Generate Activations."""

from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.list_store import ListActivationStore
from sparse_autoencoder.src_data.datasets.dummy import create_dummy_dataloader
from sparse_autoencoder.train.generate_activations import generate_activations


def test_activations_generated() -> None:
    """Check that activations are added to the store."""
    store = ListActivationStore()
    model = HookedTransformer.from_pretrained("tiny-stories-1M")

    num_samples = 10
    batch_size = 2
    dataloader = create_dummy_dataloader(num_samples, batch_size)

    generate_activations(
        model=model,
        layer=1,
        hook_name="blocks.1.mlp.hook_post",
        store=store,
        dataloader=dataloader,
        num_items=2,
    )

    assert len(store) >= 2
