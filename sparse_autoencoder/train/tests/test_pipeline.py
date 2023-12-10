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
from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    ParameterUpdateResults,
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


class TestTrainAutoencoder:
    """Test the train_autoencoder method."""

    def test_learned_activations_fired_count(self, pipeline_fixture: Pipeline) -> None:
        """Test that the learned activations fired count is updated correctly."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)
        fired_count = pipeline_fixture.train_autoencoder(store, store_size)

        assert (
            fired_count.max().item() <= store_size
        ), "Fired count should not be greater than sample size."

        assert fired_count.min().item() >= 0, "Fired count should not be negative."

        assert fired_count.sum().item() > 0, "Some neurons should have fired."

    def test_learns_with_backwards_pass(self, pipeline_fixture: Pipeline) -> None:
        """Test that the autoencoder learns with a backwards pass."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)
        pipeline_fixture.train_autoencoder(store, store_size)
        model = pipeline_fixture.autoencoder

        # Get the weights before training
        weights_before = model.encoder.weight.clone().detach()

        # Train the model
        pipeline_fixture.train_autoencoder(store, store_size)

        # Check that the weights have changed
        assert not torch.allclose(
            weights_before, model.encoder.weight
        ), "Weights should have changed after training."


class TestUpdateParameters:
    """Test the update_parameters method."""

    def test_weights_biases_changed(self, pipeline_fixture: Pipeline) -> None:
        """Test that the weights and biases have changed after training."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)
        pipeline_fixture.train_autoencoder(store, store_size)

        # Get the weights and biases before training
        encoder_weight_before = pipeline_fixture.autoencoder.encoder.weight.clone().detach()
        encoder_bias_before = pipeline_fixture.autoencoder.encoder.bias.clone().detach()
        decoder_weight_before = pipeline_fixture.autoencoder.decoder.weight.clone().detach()

        # Update the parameters
        dead_neuron_indices = torch.tensor([1, 2], dtype=torch.int64)
        pipeline_fixture.update_parameters(
            ParameterUpdateResults(
                dead_neuron_indices=dead_neuron_indices,
                dead_encoder_weight_updates=torch.zeros_like(
                    encoder_weight_before[dead_neuron_indices], dtype=torch.float
                ),
                dead_encoder_bias_updates=torch.zeros_like(
                    encoder_bias_before[dead_neuron_indices], dtype=torch.float
                ),
                dead_decoder_weight_updates=torch.zeros_like(
                    decoder_weight_before[:, dead_neuron_indices], dtype=torch.float
                ),
            )
        )

        # Check the weights and biases have changed for the dead neuron idx only
        assert not torch.allclose(
            encoder_weight_before[dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.weight[dead_neuron_indices],
        ), "Encoder weights should have changed after training."
        assert torch.allclose(
            encoder_weight_before[~dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.weight[~dead_neuron_indices],
        ), "Encoder weights should not have changed after training."

        assert not torch.allclose(
            encoder_bias_before[dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.bias[dead_neuron_indices],
        ), "Encoder biases should have changed after training."
        assert torch.allclose(
            encoder_bias_before[~dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.bias[~dead_neuron_indices],
        ), "Encoder biases should not have changed after training."

        assert not torch.allclose(
            decoder_weight_before[:, dead_neuron_indices],
            pipeline_fixture.autoencoder.decoder.weight[:, dead_neuron_indices],
        ), "Decoder weights should have changed after training."
        assert torch.allclose(
            decoder_weight_before[:, ~dead_neuron_indices],
            pipeline_fixture.autoencoder.decoder.weight[:, ~dead_neuron_indices],
        ), "Decoder weights should not have changed after training."

    def test_optimizer_state_changed(self, pipeline_fixture: Pipeline) -> None:
        """Test that the optimizer state has changed after training."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)
        pipeline_fixture.train_autoencoder(store, store_size)

        # Set the optimizer state to all 1s
        optimizer = pipeline_fixture.optimizer
        model = pipeline_fixture.autoencoder
        optimizer.state[model.encoder.weight]["exp_avg"] = torch.ones_like(
            optimizer.state[model.encoder.weight]["exp_avg"], dtype=torch.float
        )
        optimizer.state[model.encoder.weight]["exp_avg_sq"] = torch.ones_like(
            optimizer.state[model.encoder.weight]["exp_avg_sq"], dtype=torch.float
        )

        # Update the parameters
        dead_neuron_indices = torch.tensor([1, 2], dtype=torch.int64)
        pipeline_fixture.update_parameters(
            ParameterUpdateResults(
                dead_neuron_indices=dead_neuron_indices,
                dead_encoder_weight_updates=torch.zeros_like(
                    pipeline_fixture.autoencoder.encoder.weight[dead_neuron_indices],
                    dtype=torch.float,
                ),
                dead_encoder_bias_updates=torch.zeros_like(
                    pipeline_fixture.autoencoder.encoder.bias[dead_neuron_indices],
                    dtype=torch.float,
                ),
                dead_decoder_weight_updates=torch.zeros_like(
                    pipeline_fixture.autoencoder.decoder.weight[:, dead_neuron_indices],
                    dtype=torch.float,
                ),
            )
        )

        # Check the optimizer state has changed
        assert not torch.allclose(
            optimizer.state[model.encoder.weight]["exp_avg"][dead_neuron_indices],
            torch.ones_like(
                optimizer.state[model.encoder.weight]["exp_avg"][dead_neuron_indices],
                dtype=torch.float,
            ),
        ), "Optimizer dead neuron state should have changed after training."

        assert torch.allclose(
            optimizer.state[model.encoder.weight]["exp_avg"][~dead_neuron_indices],
            torch.ones_like(
                optimizer.state[model.encoder.weight]["exp_avg"][~dead_neuron_indices],
                dtype=torch.float,
            ),
        ), "Optimizer non-dead neuron state should not have changed after training."
