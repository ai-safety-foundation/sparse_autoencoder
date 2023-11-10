"""Base class for metrics."""
from abc import ABC, abstractmethod
from typing import TypedDict

from jaxtyping import Float
from torch import Tensor
from torch.optim import Optimizer

from sparse_autoencoder.autoencoder.model import SparseAutoencoder


class MetricArgs(TypedDict):
    """Class to hold arguments to metrics, contains everything needed to compute any metric."""

    step: int
    batch: Float[Tensor, "batch input_activations"]
    reconstruction_loss_mse: Float[Tensor, " item"]
    l1_loss: Float[Tensor, " item"]
    autoencoder: SparseAutoencoder
    optimizer: Optimizer
    learned_activations: Float[Tensor, "batch learned_activations"]
    reconstructed_activations: Float[Tensor, "batch input_activations"]


class Metric(ABC):
    """Base class for metrics."""

    @abstractmethod
    def compute_and_log(self, args: MetricArgs) -> None:
        """Compute the metric value."""
