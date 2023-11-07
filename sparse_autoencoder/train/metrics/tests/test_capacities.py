import torch

from sparse_autoencoder.train.metrics.capacity import (
    calc_capacities, wandb_capacities_histogram)


def test_calc_capacities() -> None:
    orthogonal_activations = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    identical_activations = torch.tensor([[-0.8, -0.8, -0.8], [-0.8, -0.8, -0.8], [-0.8, -0.8, -0.8]])
    intermediate_activations = torch.tensor([[1.0, 0.0, 0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    
    orthogonal_capacities = calc_capacities(orthogonal_activations)
    identical_capacities = calc_capacities(identical_activations)
    intermediate_capacities = calc_capacities(intermediate_activations)
    
    assert torch.allclose(orthogonal_capacities, torch.tensor([1.0, 1.0, 1.0])), "Orthogonal features should have capacity 1.0."
    assert torch.allclose(identical_capacities, torch.ones(3) / 3), "Identical features should have capacity 1/3."
    assert torch.allclose(intermediate_capacities, torch.tensor([2/3, 2/3, 1.0])), "Capacity calculation is incorrect."
    

def test_wandb_capacity_histogram() -> None:
    """Check the Weights & Biases Histogram is created correctly."""
    capacities = torch.tensor([0.5, 0.1, 1, 1, 1])
    res = wandb_capacities_histogram(capacities)

    assert res.histogram == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3], "Histogram is incorrect."