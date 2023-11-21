"""Capacity metrics for sets of learned features."""
import einops
import numpy as np
from numpy import histogram
from numpy.typing import NDArray
import torch
import wandb

from sparse_autoencoder.tensor_types import LearnedActivationBatch, TrainBatchStatistic


def calc_capacities(features: LearnedActivationBatch) -> TrainBatchStatistic:
    """Calculate capacities.

    Measure the capacity of a set of features as defined in [Polysemanticity and Capacity in Neural Networks](https://arxiv.org/pdf/2210.01892.pdf).

    Capacity is intuitively measuring the 'proportion of a dimension' assigned to a feature.
    Formally it's the ratio of the squared dot product of a feature with itself to the sum of its
    squared dot products of all features.

    If the features are orthogonal, the capacity is 1. If they are all the same, the capacity is
    1/n.

    Example:
        >>> import torch
        >>> orthogonal_features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        >>> orthogonal_caps = calc_capacities(orthogonal_features)
        >>> orthogonal_caps
        tensor([1., 1., 1.])

    Args:
        features: A collection of features.

    Returns:
        A 1D tensor of capacities, where each element is the capacity of the corresponding feature.
    """
    squared_dot_products = (
        einops.einsum(
            features, features, "n_feats1 feat_dim, n_feats2 feat_dim -> n_feats1 n_feats2"
        )
        ** 2
    )
    sum_of_sq_dot = squared_dot_products.sum(dim=-1)
    return torch.diag(squared_dot_products) / sum_of_sq_dot


def wandb_capacities_histogram(
    capacities: TrainBatchStatistic,
) -> wandb.Histogram:
    """Create a W&B histogram of the capacities.

    This can be logged with Weights & Biases using e.g. `wandb.log({"capacities_histogram":
    wandb_capacities_histogram(capacities)})`.

    Args:
        capacities: Capacity of each feature. Can be calculated using :func:`calc_capacities`.

    Returns:
        Weights & Biases histogram for logging with `wandb.log`.
    """
    numpy_capacities: NDArray[np.float_] = capacities.detach().cpu().numpy()

    bins, values = histogram(numpy_capacities, bins=20, range=(0, 1))
    return wandb.Histogram(np_histogram=(bins, values))
