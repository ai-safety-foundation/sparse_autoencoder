"""Test the pipeline module."""

import pytest
import torch
from transformer_lens import HookedTransformer

from sparse_autoencoder import (
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    Pipeline,
    SparseAutoencoder,
)
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.source_data.mock_dataset import MockDataset


@pytest.fixture()
def pipeline_fixture() -> Pipeline:
    """Fixture to create a Pipeline instance for testing."""
    device = torch.device("cpu")
    src_model = HookedTransformer.from_pretrained("tiny-stories-1M", device=device)
    autoencoder = SparseAutoencoder(
        src_model.cfg.d_model,
        src_model.cfg.d_model * 2,
    )
    loss = LossReducer(
        LearnedActivationsL1Loss(
            l1_coefficient=0.001,
        ),
        L2ReconstructionLoss(),
    )
    optimizer = AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
    )
    source_data = MockDataset(context_size=100)

    return Pipeline(
        activation_resampler=None,
        autoencoder=autoencoder,
        cache_name="blocks.0.hook_mlp_out",
        layer=0,
        loss=loss,
        optimizer=optimizer,
        source_dataset=source_data,
        source_model=src_model,
        source_data_batch_size=10,
    )


class TestGenerateActivations:
    """Test the generate_activations method."""

    def test_generates_store(self, pipeline_fixture: Pipeline) -> None:
        """Test that generate_activations generates a store."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)
        assert isinstance(
            store, TensorActivationStore
        ), "Store must be a TensorActivationStore instance"
        assert len(store) == store_size, "Store size should match the specified size"

    def test_store_has_unique_items(self, pipeline_fixture: Pipeline) -> None:
        """Test that each item from the store iterable is unique."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)

        # Get the number of unique activations generated
        activations = list(iter(store))
        activations_tensor = torch.stack(activations)
        unique_activations = activations_tensor.unique(dim=0)

        # There can be a few non-unique items (e.g. from beginnings of sentences).
        expected_min_length = len(store) * 0.9

        assert len(unique_activations) >= expected_min_length, "Store items should be unique"

    def test_two_runs_generate_different_activations(self, pipeline_fixture: Pipeline) -> None:
        """Test that two runs of generate_activations generate different activations."""
        store_size: int = 1000
        store1 = pipeline_fixture.generate_activations(store_size)
        store2 = pipeline_fixture.generate_activations(store_size)

        # Check they are different
        store1_tensor = torch.stack(list(iter(store1)))
        store2_tensor = torch.stack(list(iter(store2)))
        assert not torch.allclose(store1_tensor, store2_tensor), "Activations should be different"
