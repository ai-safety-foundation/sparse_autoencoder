"""Tests for the resample_neurons module."""
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
