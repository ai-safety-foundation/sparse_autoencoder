"""Loss Function Tests."""
import torch

from sparse_autoencoder.autoencoder.loss import loss


def test_loss() -> None:
    """Test loss against a non-vectorised approach."""
    input_activations: list[float] = [3.0, 4]
    learned_activations: list[float] = [2.0, -3]
    output_activations: list[float] = [1.0, 5]
    l1_coefficient = 0.5

    squared_errors: float = 0.0
    for i, o in zip(input_activations, output_activations):
        squared_errors += (i - o) ** 2
    mse = squared_errors / len(input_activations)

    l1_penalty: float = 0.0
    for l in learned_activations:
        l1_penalty += abs(l) * l1_coefficient

    expected: float = mse + l1_penalty

    result = loss(
        torch.tensor(input_activations),
        torch.tensor(learned_activations),
        torch.tensor(output_activations),
        torch.tensor(l1_coefficient),
    )

    assert torch.allclose(result, torch.tensor([expected]))
