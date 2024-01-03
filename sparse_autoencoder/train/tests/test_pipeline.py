"""Test the pipeline module."""
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

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
from sparse_autoencoder.activation_resampler.activation_resampler import ActivationResampler
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.metrics.abstract_metric import MetricResult
from sparse_autoencoder.metrics.validate.abstract_validate_metric import (
    AbstractValidationMetric,
    ValidationMetricData,
)
from sparse_autoencoder.source_data.mock_dataset import MockDataset


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def pipeline_fixture() -> Pipeline:
    """Fixture to create a Pipeline instance for testing."""
    device = torch.device("cpu")
    src_model = HookedTransformer.from_pretrained("tiny-stories-1M", device=device)
    autoencoder = SparseAutoencoder(
        src_model.cfg.d_model, int(src_model.cfg.d_model * 2), n_components=2
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
        has_components_dim=True,
    )
    source_data = MockDataset(context_size=10)
    activation_resampler = ActivationResampler(n_learned_features=autoencoder.n_learned_features)

    return Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        cache_names=["blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"],
        layer=1,
        loss=loss,
        optimizer=optimizer,
        source_dataset=source_data,
        source_model=src_model,
        source_data_batch_size=10,
    )


class TestInit:
    """Test the init method."""

    @pytest.mark.integration_test()
    def test_source_data_iterator_stateful(self, pipeline_fixture: Pipeline) -> None:
        """Test that the source data iterator is stateful."""
        iterator = pipeline_fixture.source_data

        sample1 = next(iterator)["input_ids"]
        sample2 = next(iterator)["input_ids"]

        assert not torch.allclose(sample1, sample2), "Source data iterator should be stateful."


class TestGenerateActivations:
    """Test the generate_activations method."""

    @pytest.mark.integration_test()
    def test_generates_store(self, pipeline_fixture: Pipeline) -> None:
        """Test that generate_activations generates a store."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size)
        assert isinstance(
            store, TensorActivationStore
        ), "Store must be a TensorActivationStore instance"
        assert len(store) == store_size, "Store size should match the specified size"

    @pytest.mark.integration_test()
    def test_store_has_unique_items(self, pipeline_fixture: Pipeline) -> None:
        """Test that each item from the store iterable is unique."""
        store_size: int = 1000
        store = pipeline_fixture.generate_activations(store_size // 2)
        store2 = pipeline_fixture.generate_activations(store_size // 2)

        # Get the number of unique activations generated
        activations = list(iter(store)) + list(iter(store2))
        activations_tensor = torch.stack(activations)
        unique_activations = activations_tensor.unique(dim=0)

        # There can be a few non-unique items (e.g. from beginnings of sentences).
        expected_min_length = len(store) * 0.9

        assert len(unique_activations) >= expected_min_length, "Store items should be unique"

    @pytest.mark.integration_test()
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

    @pytest.mark.integration_test()
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

    @pytest.mark.integration_test()
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

    @pytest.mark.integration_test()
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
            [
                ParameterUpdateResults(
                    dead_neuron_indices=dead_neuron_indices,
                    dead_encoder_weight_updates=torch.zeros_like(
                        encoder_weight_before[0, dead_neuron_indices], dtype=torch.float
                    ),
                    dead_encoder_bias_updates=torch.zeros_like(
                        encoder_bias_before[0, dead_neuron_indices], dtype=torch.float
                    ),
                    dead_decoder_weight_updates=torch.zeros_like(
                        decoder_weight_before[0, :, dead_neuron_indices], dtype=torch.float
                    ),
                ),
                ParameterUpdateResults(
                    dead_neuron_indices=torch.zeros((0), dtype=torch.int64),
                    dead_encoder_weight_updates=torch.zeros((0, 64), dtype=torch.float),
                    dead_encoder_bias_updates=torch.zeros((0), dtype=torch.float),
                    dead_decoder_weight_updates=torch.zeros((64, 0), dtype=torch.float),
                ),
            ],
        )

        # Check the weights and biases have changed for the dead neuron idx only
        assert not torch.allclose(
            encoder_weight_before[0, dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.weight[0, dead_neuron_indices],
        ), "Encoder weights should have changed after training."
        assert torch.allclose(
            encoder_weight_before[0, ~dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.weight[0, ~dead_neuron_indices],
        ), "Encoder weights should not have changed after training."

        assert not torch.allclose(
            encoder_bias_before[0, dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.bias[0, dead_neuron_indices],
        ), "Encoder biases should have changed after training."
        assert torch.allclose(
            encoder_bias_before[0, ~dead_neuron_indices],
            pipeline_fixture.autoencoder.encoder.bias[0, ~dead_neuron_indices],
        ), "Encoder biases should not have changed after training."

        assert not torch.allclose(
            decoder_weight_before[0, :, dead_neuron_indices],
            pipeline_fixture.autoencoder.decoder.weight[0, :, dead_neuron_indices],
        ), "Decoder weights should have changed after training."
        assert torch.allclose(
            decoder_weight_before[0, :, ~dead_neuron_indices],
            pipeline_fixture.autoencoder.decoder.weight[0, :, ~dead_neuron_indices],
        ), "Decoder weights should not have changed after training."

    @pytest.mark.integration_test()
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
            [
                ParameterUpdateResults(
                    dead_neuron_indices=dead_neuron_indices,
                    dead_encoder_weight_updates=torch.zeros_like(
                        pipeline_fixture.autoencoder.encoder.weight[0, dead_neuron_indices],
                        dtype=torch.float,
                    ),
                    dead_encoder_bias_updates=torch.zeros_like(
                        pipeline_fixture.autoencoder.encoder.bias[0, dead_neuron_indices],
                        dtype=torch.float,
                    ),
                    dead_decoder_weight_updates=torch.zeros_like(
                        pipeline_fixture.autoencoder.decoder.weight[0, :, dead_neuron_indices],
                        dtype=torch.float,
                    ),
                )
            ]
        )

        # Check the optimizer state has changed
        assert not torch.allclose(
            optimizer.state[model.encoder.weight]["exp_avg"][0, dead_neuron_indices],
            torch.ones_like(
                optimizer.state[model.encoder.weight]["exp_avg"][0, dead_neuron_indices],
                dtype=torch.float,
            ),
        ), "Optimizer dead neuron state should have changed after training."

        assert torch.allclose(
            optimizer.state[model.encoder.weight]["exp_avg"][0, ~dead_neuron_indices],
            torch.ones_like(
                optimizer.state[model.encoder.weight]["exp_avg"][0, ~dead_neuron_indices],
                dtype=torch.float,
            ),
        ), "Optimizer non-dead neuron state should not have changed after training."


class TestValidateSAE:
    """Test the validate_sae method."""

    @pytest.mark.integration_test()
    def test_validation_loss_calculated(self, pipeline_fixture: Pipeline) -> None:
        """Test that the validation loss numbers are calculated."""

        # Create a dummy metric, so we can retrieve the stored data afterwards
        class StoreValidationMetric(AbstractValidationMetric):
            """Dummy metric to store the data."""

            data: ValidationMetricData | None

            def calculate(self, data: ValidationMetricData) -> list[MetricResult]:
                """Store the data."""
                self.data = data
                return []

        dummy_metric = StoreValidationMetric()
        pipeline_fixture.metrics.validation_metrics.append(dummy_metric)

        # Run the validation loop
        store_size: int = 100
        pipeline_fixture.generate_activations(store_size)
        pipeline_fixture.validate_sae(store_size)

        # Check the loss is created
        assert (
            dummy_metric.data is not None
        ), "Dummy metric should have stored the data from the validation loop."
        assert (
            dummy_metric.data.source_model_loss is not None
        ), "Source model loss should be calculated."
        assert (
            dummy_metric.data.source_model_loss_with_reconstruction is not None
        ), "Source model loss with reconstruction should be calculated."
        assert (
            dummy_metric.data.source_model_loss_with_zero_ablation is not None
        ), "Source model loss with zero ablation should be calculated."


class TestSaveCheckpoint:
    """Test the save_checkpoint method."""

    @pytest.mark.integration_test()
    def test_saves_locally(self, pipeline_fixture: Pipeline) -> None:
        """Test that the save_checkpoint method saves the checkpoint locally."""
        saved_checkpoint: Path = pipeline_fixture.save_checkpoint()
        assert saved_checkpoint.exists(), "Checkpoint file should exist."

    @pytest.mark.integration_test()
    def test_saves_final(self, pipeline_fixture: Pipeline) -> None:
        """Test that the save_checkpoint method saves the final checkpoint."""
        saved_checkpoint: Path = pipeline_fixture.save_checkpoint(is_final=True)
        assert (
            "final.pt" in saved_checkpoint.name
        ), "Checkpoint file should be named '<run_name>_final.pt'."


class TestRunPipeline:
    """Test the run_pipeline method."""

    @pytest.mark.integration_test()
    def test_run_pipeline_calls_all_methods(self, pipeline_fixture: Pipeline) -> None:
        """Test that the run_pipeline method calls all the other methods."""
        pipeline_fixture.validate_sae = MagicMock(spec=Pipeline.validate_sae)  # type: ignore
        pipeline_fixture.save_checkpoint = MagicMock(spec=Pipeline.save_checkpoint)  # type: ignore
        pipeline_fixture.activation_resampler.step_resampler = MagicMock(  # type: ignore
            spec=ActivationResampler.step_resampler, return_value=None
        )

        store_size = 1000
        context_size = pipeline_fixture.source_dataset.context_size
        train_batch_size = store_size // context_size

        total_loops = 5
        validate_expected_calls = 2
        checkpoint_expected_calls = 6  # Includes final

        pipeline_fixture.run_pipeline(
            train_batch_size=train_batch_size,
            max_store_size=store_size,
            max_activations=store_size * 5,
            validation_n_activations=store_size,
            validate_frequency=store_size * (total_loops // validate_expected_calls),
            checkpoint_frequency=store_size,
        )

        # Check the number of calls
        assert (
            pipeline_fixture.validate_sae.call_count == validate_expected_calls
        ), f"Validate should have been called {validate_expected_calls} times."

        assert (
            pipeline_fixture.save_checkpoint.call_count == checkpoint_expected_calls
        ), f"Checkpoint should have been called {checkpoint_expected_calls} times."

        assert (pipeline_fixture.activation_resampler) is not None
        assert (
            pipeline_fixture.activation_resampler.step_resampler.call_count == total_loops
        ), f"Resampler should have been called {total_loops} times."
