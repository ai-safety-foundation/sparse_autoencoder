"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module, Parameter

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

    @property
    def reset_optimizer_parameter_details(self) -> list[tuple[Parameter, int]]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """
        return (
            self.encoder.reset_optimizer_parameter_details
            + self.decoder.reset_optimizer_parameter_details
        )

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
