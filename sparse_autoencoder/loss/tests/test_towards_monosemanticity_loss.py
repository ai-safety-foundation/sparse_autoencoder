"""Test the loss function from the Towards Monosemanticity paper."""
import torch

from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
from sparse_autoencoder.loss.mse_reconstruction_loss import MSEReconstructionLoss
from sparse_autoencoder.loss.reducer import LossReducer


class TestTowardsMonosemanticityLoss:
    """Test the loss function from the Towards Monosemanticity paper."""

    def test_loss(self) -> None:
        """Test loss against a non-vectorised approach."""
        # Calculate the expected loss with a non-vectorised approach
        input_activations: list[float] = [3.0, 4]
        learned_activations: list[float] = [2.0, -3]
        output_activations: list[float] = [1.0, 5]
        l1_coefficient = 0.5

        squared_errors: float = 0.0
        for i, o in zip(input_activations, output_activations, strict=True):
            squared_errors += (i - o) ** 2
        mse = squared_errors / len(input_activations)

        l1_penalty: float = 0.0
        for neuron in learned_activations:
            l1_penalty += abs(neuron) * l1_coefficient

        expected: float = mse + l1_penalty

        # Compare against the actual loss function
        loss = LossReducer(
            MSEReconstructionLoss(),
            LearnedActivationsL1Loss(l1_coefficient),
        )

        result, _metrics = loss(
            torch.tensor(input_activations).unsqueeze(0),
            torch.tensor(learned_activations).unsqueeze(0),
            torch.tensor(output_activations).unsqueeze(0),
        )

        assert torch.allclose(result, torch.tensor(expected))
