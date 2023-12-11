"""Abstract Outer Bias.

This can be extended to create e.g. a pre-encoder and post-decoder bias.
"""
from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module, Parameter

from sparse_autoencoder.tensor_types import Axis


class AbstractOuterBias(Module, ABC):
    """Abstract Pre-Encoder or Post-Decoder Bias Module."""

    @property
    @abstractmethod
    def bias(self) -> Float[Parameter, Axis.INPUT_OUTPUT_FEATURE]:
        """Bias.

        May be a reference to a bias parameter in the parent module, if using e.g. a tied bias.
        """

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Resulting activations.
        """
