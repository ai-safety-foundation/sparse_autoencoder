"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module

from sparse_autoencoder.autoencoder.components.abstract_decoder import AbstractDecoder
from sparse_autoencoder.autoencoder.components.abstract_encoder import AbstractEncoder
from sparse_autoencoder.autoencoder.components.abstract_outer_bias import AbstractOuterBias
from sparse_autoencoder.tensor_types import Axis


class AbstractAutoencoder(Module, ABC):
    """Abstract Sparse Autoencoder Model."""

    @property
    @abstractmethod
    def encoder(self) -> AbstractEncoder:
        """Encoder."""

    @property
    @abstractmethod
    def decoder(self) -> AbstractDecoder:
        """Decoder."""

    @property
    @abstractmethod
    def pre_encoder_bias(self) -> AbstractOuterBias:
        """Pre-encoder bias."""

    @property
    @abstractmethod
    def post_decoder_bias(self) -> AbstractOuterBias:
        """Post-decoder bias."""

    @abstractmethod
    def forward(
        self,
        x: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
    ) -> tuple[
        Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)],
        Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)],
    ]:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Tuple of learned activations and decoded activations.
        """

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters."""
