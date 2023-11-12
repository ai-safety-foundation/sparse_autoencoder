"""Tests for the resample_neurons module."""
import copy

from jaxtyping import Float, Int
import pytest
import torch
from torch import Tensor

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.train.resample_neurons import (
    assign_sampling_probabilities,
    compute_loss_and_get_activations,
    get_dead_neuron_indices,
    renormalize_and_scale,
    resample_dead_neurons,
    sample_input,
)
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime


DEFAULT_N_ITEMS: int = 100
DEFAULT_N_INPUT_NEURONS: int = 5


@pytest.fixture()
def input_activations_fixture(
    items: int = DEFAULT_N_ITEMS, neurons: int = DEFAULT_N_INPUT_NEURONS
) -> Tensor:
    """Create a dummy input activations tensor."""
    return torch.rand((items, neurons))


@pytest.fixture()
def activation_store_fixture(
    input_activations_fixture: Tensor,
) -> ActivationStore:
    """Create a dummy activation store.

    Creates a store for use in tests, pre-populated with mock data.
    """
    max_items = input_activations_fixture.shape[0]
    max_neurons = input_activations_fixture.shape[1]
    store = TensorActivationStore(max_items, max_neurons)

    store.extend(input_activations_fixture)

    return store


@pytest.fixture()
def sweep_parameters_fixture() -> SweepParametersRuntime:
    """Create a dummy sweep parameters object."""
    return SweepParametersRuntime()


@pytest.fixture()
def autoencoder_model_fixture(n_input_neurons: int = DEFAULT_N_INPUT_NEURONS) -> SparseAutoencoder:
    """Create a dummy autoencoder model."""
    return SparseAutoencoder(
        n_input_neurons,
        n_learned_features=n_input_neurons * 4,
        geometric_median_dataset=torch.zeros((n_input_neurons), dtype=torch.float32),
    )


class TestGetDeadNeuronIndices:
    """Tests for get_dead_neuron_indices."""

    @pytest.mark.parametrize(
        ("neuron_activity", "threshold", "expected_indices"),
        [
            (torch.tensor([1, 0, 3, 9, 0]), 0, torch.tensor([1, 4])),
            (torch.tensor([1, 2, 3, 4, 5]), 0, torch.tensor([])),
            (torch.tensor([1, 0, 3, 9, 0]), 1, torch.tensor([0, 1, 4])),
            (torch.tensor([1, 2, 3, 4, 5]), 1, torch.tensor([0])),
        ],
    )
    def test_get_dead_neuron_indices(
        self, neuron_activity: Tensor, threshold: int, expected_indices: Tensor
    ) -> None:
        """Test the dead neuron indices match manually created examples."""
        res = get_dead_neuron_indices(neuron_activity, threshold)
        assert torch.equal(res, expected_indices), f"Expected {expected_indices}, got {res}"


class TestComputeLossAndGetActivations:
    """Tests for compute_loss_and_get_activations."""

    def test_gets_loss_and_correct_activations(
        self,
        activation_store_fixture: ActivationStore,
        autoencoder_model_fixture: SparseAutoencoder,
        sweep_parameters_fixture: SweepParametersRuntime,
        input_activations_fixture: Tensor,
    ) -> None:
        """Test it gets loss and also returns the input activations."""
        loss, input_activations = compute_loss_and_get_activations(
            activation_store_fixture,
            autoencoder_model_fixture,
            sweep_parameters_fixture,
            DEFAULT_N_ITEMS,
        )

        assert isinstance(loss, Tensor)
        assert isinstance(input_activations, Tensor)
        assert loss.shape == (DEFAULT_N_ITEMS,)
        assert input_activations.shape == (DEFAULT_N_ITEMS, DEFAULT_N_INPUT_NEURONS)

        # Check that the activations are the same as the input data
        assert torch.equal(input_activations, input_activations_fixture)

    def test_more_items_than_in_store_error(
        self,
        activation_store_fixture: ActivationStore,
        autoencoder_model_fixture: SparseAutoencoder,
        sweep_parameters_fixture: SweepParametersRuntime,
    ) -> None:
        """Test that an error is raised if there are more items than in the store."""
        with pytest.raises(
            ValueError,
            match=r"Cannot get \d+ items from the store, as only \d+ were available.",
        ):
            compute_loss_and_get_activations(
                activation_store_fixture,
                autoencoder_model_fixture,
                sweep_parameters_fixture,
                DEFAULT_N_ITEMS + 1,
            )


class TestAssignSamplingProbabilities:
    """Tests for assign_sampling_probabilities."""

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
        probabilities = assign_sampling_probabilities(loss)

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
            sampled_input = sample_input(probabilities, input_activations, 1)

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
        sampled_input = sample_input(probabilities, input_activations, 0)
        assert sampled_input.shape == (0, 2), "Should return an empty tensor"

    def test_sample_input_raises_value_error(self) -> None:
        """Test that ValueError is raised on length miss-match."""
        probabilities = torch.tensor([0.1, 0.2, 0.7])
        input_activations = torch.tensor([[1.0, 2], [3, 4], [5, 6]])
        num_samples = 4  # More than the number of input activations

        with pytest.raises(
            ValueError, match=r"Cannot sample \d+ inputs from \d+ input activations."
        ):
            sample_input(probabilities, input_activations, num_samples)


class TestRenormalizeAndScale:
    """Tests for renormalize_and_scale."""

    @staticmethod
    def calculate_expected_output(
        sampled_input: Float[Tensor, "dead_neuron input_feature"],
        neuron_activity: Int[Tensor, " learned_features"],
        encoder_weight: Float[Tensor, "learned_feature input_feature"],
    ) -> Float[Tensor, "dead_neuron input_feature"]:
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
        sampled_input = torch.tensor([[3.0, 4.0]])
        neuron_activity = torch.tensor([1, 0, 1, 0, 1, 1])
        encoder_weight = torch.ones((6, 2))

        rescaled_input = renormalize_and_scale(sampled_input, neuron_activity, encoder_weight)

        expected_output = self.calculate_expected_output(
            sampled_input, neuron_activity, encoder_weight
        )

        assert torch.allclose(rescaled_input, expected_output), "Basic renormalization failed"

    def test_all_alive_neurons(self) -> None:
        """Test behavior when all neurons are alive."""
        sampled_input = torch.empty((0, 2), dtype=torch.float32)
        neuron_activity = torch.tensor([1, 4, 1, 3, 1, 1])
        encoder_weight = torch.ones((6, 2))

        rescaled_input = renormalize_and_scale(sampled_input, neuron_activity, encoder_weight)

        assert rescaled_input.shape == (0, 2), "Should return an empty tensor"


class TestResampleDeadNeurons:
    """Tests for resample_dead_neurons."""

    def test_no_changes_if_no_dead_neurons(self) -> None:
        """Check it doesn't change anything if there are no dead neurons."""
        neuron_activity = torch.ones(10, dtype=torch.int32)
        store_data = torch.rand((100, 5))
        store = TensorActivationStore(100, 5)
        store.extend(store_data)
        sweep_parameters = SweepParametersRuntime()
        model = SparseAutoencoder(5, 10, torch.rand(5))
        optimizer = torch.optim.Adam(model.parameters())

        # Run a forward pass to initialize the optimizer state
        _learned_activations, decoded_activations = model(store_data)
        loss = decoded_activations.sum()  # Dummy loss
        loss.backward()
        optimizer.step()

        current_parameters = model.state_dict()
        resample_dead_neurons(neuron_activity, store, model, sweep_parameters, 100)
        updated_parameters = model.state_dict()

        for key in current_parameters:
            assert torch.equal(current_parameters[key], updated_parameters[key])

    def test_updates_a_dead_neuron_parameters(self) -> None:
        """Check it updates a dead neuron's parameters."""
        neuron_activity = torch.ones(10, dtype=torch.int32)
        dead_neuron_idx = 3
        neuron_activity[dead_neuron_idx] = 0
        store_data = torch.rand((100, 5))
        store = TensorActivationStore(100, 5)
        store.extend(store_data)
        sweep_parameters = SweepParametersRuntime()
        model = SparseAutoencoder(5, 10, torch.rand(5))
        optimizer = torch.optim.Adam(model.parameters())

        # Run a forward pass to initialize the optimizer state
        _learned_activations, decoded_activations = model(store_data)
        loss = decoded_activations.sum()  # Dummy loss
        loss.backward()
        optimizer.step()

        current_parameters = copy.deepcopy(model.state_dict())
        resample_dead_neurons(neuron_activity, store, model, sweep_parameters, 100)
        updated_parameters = model.state_dict()

        for key in current_parameters:
            if "tied_bias" in key or "TiedBias" in key:
                assert torch.equal(current_parameters[key], updated_parameters[key])

            else:
                assert not torch.equal(
                    current_parameters[key], updated_parameters[key]
                ), f"Parameter {key} should have changed."
