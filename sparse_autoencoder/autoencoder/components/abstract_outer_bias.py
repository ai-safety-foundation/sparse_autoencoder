"""Abstract Outer Bias.

This can be extended to create e.g. a pre-encoder and post-decoder bias.
"""
from abc import ABC, abstractmethod

from torch.nn import Module

from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    InputOutputActivationVector,
)


class AbstractOuterBias(Module, ABC):
    """Abstract Pre-Encoder or Post-Decoder Bias Module."""

    @property
    @abstractmethod
    def bias(self) -> InputOutputActivationVector:
        """Bias.

        May be a reference to a bias parameter in the parent module, if using e.g. a tied bias.
        """

    @abstractmethod
    def forward(
        self,
        x: InputOutputActivationBatch,
    ) -> InputOutputActivationBatch:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Resulting activations.
        """
