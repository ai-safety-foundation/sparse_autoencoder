"""Feature density metrics & histogram."""

import einops
from jaxtyping import Float
from numpy import histogram
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import wandb


def calc_feature_density(
    activations: Float[Tensor, "sample activation"], threshold: float = 0.001
) -> Float[Tensor, " activation"]:
    """Count how many times each feature was active.

    Percentage of samples in which each feature was active (i.e. the neuron has "fired").

    Example:
        >>> import torch
        >>> activations = torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.0001]])
        >>> calc_feature_density(activations).tolist()
        [1.0, 0.5, 0.0]

    Args:
        activations: Sample of cached activations (the Autoencoder's learned features).
        threshold: Threshold for considering a feature active (i.e. the neuron has "fired"). This
            should be close to zero.

    Returns:
        Number of times each feature was active in a sample.
    """
    has_fired: Float[Tensor, "sample activation"] = torch.gt(activations, threshold).to(
        # Use float as einops requires this (64 as some features are very sparse)
        dtype=torch.float64
    )

    return einops.reduce(has_fired, "sample activation -> activation", "mean")


def wandb_feature_density_histogram(
    feature_density: Float[Tensor, " activation"],
) -> wandb.Histogram:
    """Create a W&B histogram of the feature density.

    This can be logged with Weights & Biases using e.g. `wandb.log({"feature_density_histogram":
    wandb_feature_density_histogram(feature_density)})`.

    Args:
        feature_density: Number of times each feature was active in a sample. Can be calculated
            using :func:`feature_activity_count`.

    Returns:
        Weights & Biases histogram for logging with `wandb.log`.
    """
    numpy_feature_density: NDArray[np.float_] = feature_density.detach().cpu().numpy()

    bins, values = histogram(numpy_feature_density, bins="auto")
    return wandb.Histogram(np_histogram=(bins, values))
