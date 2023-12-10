"""Tests for the resample_neurons module."""

from jaxtyping import Float, Int, Int64
import pytest
import torch
from torch import Tensor
from torch.nn import Parameter

from sparse_autoencoder.activation_resampler.activation_resampler import ActivationResampler
from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.decoded_activations_l2 import L2ReconstructionLoss
from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
from sparse_autoencoder.loss.reducer import LossReducer
from sparse_autoencoder.tensor_types import Axis


DEFAULT_N_ACTIVATIONS_STORE: int = 100
DEFAULT_N_INPUT_FEATURES: int = 3
DEFAULT_N_LEARNED_FEATURES: int = 5


@pytest.fixture()
def random_activations(
    items: int = DEFAULT_N_ACTIVATIONS_STORE, n_input_features: int = DEFAULT_N_INPUT_FEATURES
) -> Int[Tensor, Axis.LEARNT_FEATURE]:
    """Create a dummy input activations tensor."""
    return torch.rand((items, n_input_features))


@pytest.fixture()
def full_activation_store(
    random_activations: Tensor,
) -> ActivationStore:
    """Create a dummy activation store, pre-populated with data."""
    store = TensorActivationStore(DEFAULT_N_ACTIVATIONS_STORE, DEFAULT_N_INPUT_FEATURES)
    store.extend(random_activations)
    return store


@pytest.fixture()
def autoencoder_model() -> SparseAutoencoder:
    """Create a dummy autoencoder model."""
    return SparseAutoencoder(
        n_input_features=DEFAULT_N_INPUT_FEATURES,
        n_learned_features=DEFAULT_N_LEARNED_FEATURES,
    )


@pytest.fixture()
def loss_fn() -> LossReducer:
    """Loss function fixture."""
    return LossReducer(LearnedActivationsL1Loss(0.01), L2ReconstructionLoss())


@pytest.fixture()
def activation_resampler_single_item_triggers() -> ActivationResampler:
    """Activation resampler where any call to step will result in resampling."""
    return ActivationResampler(
        n_learned_features=DEFAULT_N_LEARNED_FEATURES,
        resample_interval=1,
        n_activations_activity_collate=1,
        resample_dataset_size=1,
        threshold_is_dead_portion_fires=0.0,
    )


class TestInit:
    """Tests for the activation resampler initialisation."""

    @pytest.mark.parametrize(
        ("resample_interval", "n_steps_collate", "expected_window_start"),
        [(100, 50, 50), (100, 100, 0)],
    )
    def test_neuron_activity_window_start(
        self, resample_interval: int, n_steps_collate: int, expected_window_start: int
    ) -> None:
        """Test the neuron activity window start is set correctly."""
        resampler = ActivationResampler(
            n_learned_features=10,
            resample_interval=resample_interval,
            n_activations_activity_collate=n_steps_collate,
        )

        assert resampler.neuron_activity_window_start == expected_window_start

    def test_n_steps_collate_less_0_errors(self) -> None:
        """Check it raises an error if the number of steps to collate <= 0."""
        with pytest.raises(
            ValueError,
            match=r"greater than 0",
        ):
            ActivationResampler(n_learned_features=100, n_activations_activity_collate=-1)

        with pytest.raises(
            ValueError,
            match=r"greater than 0",
        ):
            ActivationResampler(n_learned_features=100, n_activations_activity_collate=0)

    def test_threshold_is_dead_portion_error(self) -> None:
        """Check it raises if the threshold is dead portion is <0 or >1."""
        with pytest.raises(
            ValueError,
            match=r"between 0 and 1",
        ):
            ActivationResampler(n_learned_features=100, threshold_is_dead_portion_fires=-1.0)

        with pytest.raises(
            ValueError,
            match=r"between 0 and 1",
        ):
            ActivationResampler(n_learned_features=10, threshold_is_dead_portion_fires=2.0)

    def test_max_n_resamples_less_0_error(self) -> None:
        """Check it raises if the max number or resamples is less than 0."""
        with pytest.raises(
            ValueError,
            match=r"greater than 0",
        ):
            ActivationResampler(n_learned_features=100, max_n_resamples=-1)

    def test_resample_dataset_size_less_0_error(self) -> None:
        """Check it raises if the resample dataset size is less than 0."""
        with pytest.raises(
            ValueError,
            match=r"greater than 0",
        ):
            ActivationResampler(n_learned_features=100, resample_dataset_size=-1)


class TestComputeLossAndGetActivations:
    """Tests for compute_loss_and_get_activations."""

    def test_gets_loss_and_correct_activations(
        self,
        full_activation_store: ActivationStore,
        autoencoder_model: SparseAutoencoder,
        random_activations: Tensor,
    ) -> None:
        """Test it gets loss and also returns the input activations."""
        resampler = ActivationResampler(
            n_learned_features=autoencoder_model.n_learned_features,
            resample_dataset_size=DEFAULT_N_ACTIVATIONS_STORE,
        )
        loss, input_activations = resampler.compute_loss_and_get_activations(
            store=full_activation_store,
            autoencoder=autoencoder_model,
            loss_fn=L2ReconstructionLoss(),
            train_batch_size=DEFAULT_N_ACTIVATIONS_STORE,
        )

        assert isinstance(loss, Tensor)
        assert isinstance(input_activations, Tensor)
        assert loss.shape == (DEFAULT_N_ACTIVATIONS_STORE,)
        assert input_activations.shape == (DEFAULT_N_ACTIVATIONS_STORE, DEFAULT_N_INPUT_FEATURES)

        # Check that the activations are the same as the input data
        assert torch.equal(input_activations, random_activations)

    def test_more_items_than_in_store_error(
        self,
        full_activation_store: ActivationStore,
        autoencoder_model: SparseAutoencoder,
    ) -> None:
        """Test that an error is raised if there are more items than in the store."""
        with pytest.raises(
            ValueError,
            match=r"Cannot get \d+ items from the store, as only \d+ were available.",
        ):
            ActivationResampler(
                resample_dataset_size=DEFAULT_N_ACTIVATIONS_STORE + 1,
                n_learned_features=autoencoder_model.n_learned_features,
            ).compute_loss_and_get_activations(
                store=full_activation_store,
                autoencoder=autoencoder_model,
                loss_fn=L2ReconstructionLoss(),
                train_batch_size=DEFAULT_N_ACTIVATIONS_STORE + 1,
            )


class TestAssignSamplingProbabilities:
    """Test the assign sampling probabilities method."""

    @pytest.mark.parametrize(
        ("loss"),
        [
            (torch.tensor([1.0, 2.0, 3.0])),
            (torch.tensor([2.0, 3.0, 5.0])),
            (torch.tensor([0.0, 100.0])),
        ],
    )
    def test_assign_sampling_probabilities(self, loss: Tensor) -> None:
        """Test that sampling probabilities are correctly assigned based on loss."""
        probabilities = ActivationResampler.assign_sampling_probabilities(loss)

        # Compare against non-vectorized implementation
        squared_loss = [batch_item_loss.item() ** 2 for batch_item_loss in loss]
        sum_squared = sum(squared_loss)
        proportions = [item / sum_squared for item in squared_loss]
        expected_probabilities = torch.tensor(proportions)

        assert torch.allclose(
            probabilities, expected_probabilities, atol=1e-4
        ), f"Expected probabilities {expected_probabilities} but got {probabilities}"


class TestSampleInput:
    """Tests for sample_input."""

    def test_distribution(self) -> None:
        """Test that sample approximately matches a multinomial distribution."""
        torch.manual_seed(0)

        probabilities = torch.tensor([0.1, 0.2, 0.7])

        results = [0, 0, 0]
        for _ in range(10_000):
            input_activations = torch.tensor([[0.0, 0], [1, 1], [2, 2]])
            sampled_input = ActivationResampler.sample_input(probabilities, input_activations, 1)

            # Get the input activation index (the first element is also the index)
            sampled_activation_idx = sampled_input[0][0].item()

            results[int(sampled_activation_idx)] += 1

        resulting_probabilities = torch.tensor([item / sum(results) for item in results])

        assert torch.allclose(
            resulting_probabilities, probabilities, atol=1e-2
        ), f"Expected probabilities {probabilities} but got {resulting_probabilities}"

    def test_zero_probabilities(self) -> None:
        """Test where there are no dead neurons."""
        probabilities = torch.tensor([0.0, 0.0, 1.0])
        input_activations = torch.tensor([[0.0, 0], [1, 1], [2, 2]])
        sampled_input = ActivationResampler.sample_input(probabilities, input_activations, 0)
        assert sampled_input.shape == (0, 2), "Should return an empty tensor"

    def test_sample_input_raises_value_error(self) -> None:
        """Test that ValueError is raised on length miss-match."""
        probabilities = torch.tensor([0.1, 0.2, 0.7])
        input_activations = torch.tensor([[1.0, 2], [3, 4], [5, 6]])
        num_samples = 4  # More than the number of input activations

        with pytest.raises(
            ValueError, match=r"Cannot sample \d+ inputs from \d+ input activations."
        ):
            ActivationResampler.sample_input(probabilities, input_activations, num_samples)


class TestRenormalizeAndScale:
    """Tests for renormalize_and_scale."""

    @staticmethod
    def calculate_expected_output(
        sampled_input: Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)],
        neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE],
        encoder_weight: Float[
            Parameter, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]:
        """Non-vectorized approach to compare against."""
        # Initialize variables
        total_norm = 0
        alive_neurons_count = 0

        # Iterate through each neuron
        for i in range(neuron_activity.shape[0]):
            if neuron_activity[i] > 0:  # Check if the neuron is alive
                weight = encoder_weight[i]
                norm = torch.norm(weight)  # Calculate the norm of the encoder weight
                total_norm += norm
                alive_neurons_count += 1

        # Calculate the average norm for alive neurons
        average_alive_norm = total_norm / alive_neurons_count if alive_neurons_count > 0 else 0

        # Renormalize the input vector
        renormalized_input = torch.nn.functional.normalize(sampled_input, dim=-1)

        # Scale by the average norm times 0.2
        return renormalized_input * (average_alive_norm * 0.2)

    def test_basic_renormalization(self) -> None:
        """Test basic renormalization with simple inputs."""
        sampled_input: Float[
            Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
        ] = torch.tensor([[3.0, 4.0, 5.0]])
        neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE] = torch.tensor([1, 0, 1, 0, 1])
        encoder_weight: Float[
            Parameter, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
        ] = Parameter(torch.ones((DEFAULT_N_LEARNED_FEATURES, DEFAULT_N_INPUT_FEATURES)))

        rescaled_input = ActivationResampler.renormalize_and_scale(
            sampled_input, neuron_activity, encoder_weight
        )

        expected_output = self.calculate_expected_output(
            sampled_input, neuron_activity, encoder_weight
        )

        assert torch.allclose(rescaled_input, expected_output), "Basic renormalization failed"

    def test_all_alive_neurons(self) -> None:
        """Test behavior when all neurons are alive."""
        sampled_input = torch.empty((0, 2), dtype=torch.float32)
        neuron_activity = torch.tensor([1, 4, 1, 3, 1, 1])
        encoder_weight = Parameter(torch.ones((6, 2)))

        rescaled_input = ActivationResampler.renormalize_and_scale(
            sampled_input, neuron_activity, encoder_weight
        )

        assert rescaled_input.shape == (0, 2), "Should return an empty tensor"


class TestResampleDeadNeurons:
    """Tests for resample_dead_neurons."""

    def test_no_changes_if_no_dead_neurons(
        self, full_activation_store: ActivationStore, autoencoder_model: SparseAutoencoder
    ) -> None:
        """Check it doesn't change anything if there are no dead neurons."""
        neuron_activity = torch.ones(DEFAULT_N_LEARNED_FEATURES, dtype=torch.int64)
        resampler = ActivationResampler(
            resample_interval=10,
            n_activations_activity_collate=10,
            n_learned_features=autoencoder_model.n_learned_features,
            resample_dataset_size=100,
        )
        updates = resampler.step_resampler(
            batch_neuron_activity=neuron_activity,
            activation_store=full_activation_store,
            autoencoder=autoencoder_model,
            loss_fn=L2ReconstructionLoss(),
            train_batch_size=10,
        )

        assert updates is not None, "Should have updated"
        assert updates.dead_neuron_indices.numel() == 0, "Should not have any dead neurons"
        assert updates.dead_decoder_weight_updates.numel() == 0, "Should not have any dead neurons"
        assert updates.dead_encoder_weight_updates.numel() == 0, "Should not have any dead neurons"
        assert updates.dead_encoder_bias_updates.numel() == 0, "Should not have any dead neurons"

    def test_updates_a_dead_neuron_parameters(
        self, full_activation_store: ActivationStore, autoencoder_model: SparseAutoencoder
    ) -> None:
        """Check it updates a dead neuron's parameters."""
        neuron_activity = torch.ones(DEFAULT_N_LEARNED_FEATURES, dtype=torch.int64)
        dead_neuron_idx = 2
        neuron_activity[dead_neuron_idx] = 0

        # Get the current & updated parameters
        current_parameters = autoencoder_model.state_dict()
        resampler = ActivationResampler(
            resample_interval=10,
            n_activations_activity_collate=10,
            n_learned_features=autoencoder_model.n_learned_features,
            resample_dataset_size=100,
        )
        parameter_updates = resampler.step_resampler(
            batch_neuron_activity=neuron_activity,
            activation_store=full_activation_store,
            autoencoder=autoencoder_model,
            loss_fn=L2ReconstructionLoss(),
            train_batch_size=10,
        )
        assert parameter_updates is not None, "Should have updated"

        # Check the updated ones have changed
        current_dead_decoder_weights = current_parameters["_decoder._weight"][:, dead_neuron_idx]
        updated_dead_decoder_weights = parameter_updates.dead_encoder_weight_updates.squeeze()
        assert not torch.equal(
            current_dead_decoder_weights, updated_dead_decoder_weights
        ), "Dead decoder weights should have changed."

        current_dead_encoder_weights = current_parameters["_encoder._weight"][dead_neuron_idx]
        updated_dead_encoder_weights = parameter_updates.dead_encoder_weight_updates.squeeze()
        assert not torch.equal(
            current_dead_encoder_weights, updated_dead_encoder_weights
        ), "Dead encoder weights should have changed."

        current_dead_encoder_bias = current_parameters["_encoder._bias"][dead_neuron_idx]
        updated_dead_encoder_bias = parameter_updates.dead_encoder_bias_updates
        assert not torch.equal(
            current_dead_encoder_bias, updated_dead_encoder_bias
        ), "Dead encoder bias should have changed."


class TestStepResampler:
    """Tests for stepping the activation resampler."""

    @pytest.mark.parametrize(
        ("neuron_activity", "threshold", "expected_indices"),
        [
            (torch.tensor([1, 0, 3, 9, 0]), 0.0, torch.tensor([1, 4], dtype=torch.int64)),
            (
                torch.tensor([1, 2, 3, 4, 5]),
                0.0,
                torch.tensor([], dtype=torch.int64),
            ),
            (torch.tensor([1, 0, 3, 9, 0]), 0.1, torch.tensor([0, 1, 4], dtype=torch.int64)),
            (torch.tensor([1, 2, 3, 4, 5]), 0.1, torch.tensor([0], dtype=torch.int64)),
        ],
    )
    def test_gets_dead_neuron_indices(
        self,
        neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE],
        threshold: float,
        expected_indices: Tensor,
        full_activation_store: ActivationStore,
        autoencoder_model: SparseAutoencoder,
        loss_fn: LossReducer,
    ) -> None:
        """Test the dead neuron indices match manually created examples."""
        resampler = ActivationResampler(
            n_learned_features=DEFAULT_N_LEARNED_FEATURES,
            resample_interval=1,
            n_activations_activity_collate=1,
            resample_dataset_size=1,
            threshold_is_dead_portion_fires=threshold,
        )
        res = resampler.step_resampler(
            neuron_activity,
            full_activation_store,
            autoencoder_model,
            loss_fn,
            train_batch_size=10,
        )
        assert res is not None

        assert torch.allclose(
            res.dead_neuron_indices, expected_indices
        ), f"Expected {expected_indices}, got {res.dead_neuron_indices}"

    @pytest.mark.parametrize(
        (
            "max_n_resamples",
            "resample_interval",
            "total_activations_seen",
            "should_update",
            "assert_fail_message",
        ),
        [
            (2, 3, 1, False, "Shouldn't have resampled at the start"),
            (2, 3, 3, True, "Should have resampled for the first time"),
            (2, 3, 5, False, "Shouldn't have resampled in between resample intervals"),
            (2, 3, 6, True, "Should have resampled for the last time"),
            (2, 3, 5, False, "Shouldn't have resampled past max updated"),
        ],
    )
    def test_max_updates(
        self,
        *,
        max_n_resamples: int,
        resample_interval: int,
        total_activations_seen: int,
        should_update: bool,
        assert_fail_message: str,
        autoencoder_model: SparseAutoencoder,
    ) -> None:
        """Check if max_updates, resample_interval and n_steps_collate are respected."""
        # Create neuron activity to log (with one dead neuron)
        neuron_activity_batch_size_1 = torch.ones(DEFAULT_N_LEARNED_FEATURES, dtype=torch.int64)
        neuron_activity_batch_size_1[2] = 0

        resampler = ActivationResampler(
            n_learned_features=autoencoder_model.n_learned_features,
            resample_interval=resample_interval,
            max_n_resamples=max_n_resamples,
            n_activations_activity_collate=1,
            resample_dataset_size=1,
        )

        for activation_seen_count in range(1, total_activations_seen + 1):
            activation_store = TensorActivationStore(1, DEFAULT_N_INPUT_FEATURES)
            activation_store.append(torch.ones(DEFAULT_N_INPUT_FEATURES))

            updates = resampler.step_resampler(
                batch_neuron_activity=neuron_activity_batch_size_1,
                activation_store=activation_store,
                autoencoder=autoencoder_model,
                loss_fn=L2ReconstructionLoss(),
                train_batch_size=1,
            )

            if activation_seen_count == total_activations_seen:
                has_updated = updates is not None
                assert has_updated == should_update, assert_fail_message
