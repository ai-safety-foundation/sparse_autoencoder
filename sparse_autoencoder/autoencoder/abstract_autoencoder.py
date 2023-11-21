"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod

from torch.nn import Module

from sparse_autoencoder.autoencoder.components.abstract_decoder import AbstractDecoder
from sparse_autoencoder.autoencoder.components.abstract_encoder import AbstractEncoder
from sparse_autoencoder.autoencoder.components.abstract_outer_bias import AbstractOuterBias
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    LearnedActivationBatch,
)


class AbstractAutoencoder(Module, ABC):
    """Abstract Sparse Autoencoder Model."""

    @property
    @abstractmethod
    def encoder(self) -> AbstractEncoder:
        """Encoder."""
        raise NotImplementedError

    @property
    @abstractmethod
    def decoder(self) -> AbstractDecoder:
        """Decoder."""
        raise NotImplementedError

    @property
    @abstractmethod
    def pre_encoder_bias(self) -> AbstractOuterBias:
        """Pre-encoder bias."""
        raise NotImplementedError

    @property
    @abstractmethod
    def post_decoder_bias(self) -> AbstractOuterBias:
        """Post-decoder bias."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: InputOutputActivationBatch,
    ) -> tuple[
        LearnedActivationBatch,
        InputOutputActivationBatch,
    ]:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Tuple of learned activations and decoded activations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters."""
        raise NotImplementedError
