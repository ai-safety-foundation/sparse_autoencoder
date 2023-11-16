"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod
from typing import TypedDict, final

import torch
from torch.nn import Module

from sparse_autoencoder.tensor_types import (
    DecoderWeights,
    EncoderWeights,
    InputOutputActivationBatch,
    InputOutputActivationVector,
    LearnedActivationBatch,
    LearntActivationVector,
)


class AbstractEncoder(Module, ABC):
    """Abstract autoencoder module.

    Typically includes :attr:`weights` and :attr:`bias` parameters, as well as an activation
    function.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device | None:
        """Device to run the module on."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype | None:
        """Data type to use for the module."""

    @property
    @abstractmethod
    def weights(self) -> EncoderWeights:
        """Weights."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bias(self) -> LearntActivationVector | None:
        """Bias."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: InputOutputActivationBatch,
    ) -> LearnedActivationBatch:
        """Forward pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Learned activations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters."""
        raise NotImplementedError


class AbstractDecoder(Module, ABC):
    """Abstract Decoder Module.

    Typically includes just a :attr:`weights` parameter.
    """

    @property
    @abstractmethod
    def device(self) -> torch.device | None:
        """Device to run the module on."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype | None:
        """Data type to use for the module."""

    @property
    @abstractmethod
    def weights(self) -> DecoderWeights:
        """Weights."""
        raise NotImplementedError

    @property
    @abstractmethod
    def bias(self) -> InputOutputActivationVector | None:
        """Bias."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        x: LearnedActivationBatch,
    ) -> InputOutputActivationBatch:
        """Forward Pass.

        Args:
            x: Learned activations.

        Returns:
            Decoded activations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters."""
        raise NotImplementedError


class AbstractOuterBias(Module, ABC):
    """Abstract Pre-Encoder or Post-Decoder Bias Module."""

    @property
    @abstractmethod
    def device(self) -> torch.device | None:
        """Device to run the module on."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype | None:
        """Data type to use for the module."""

    @property
    @abstractmethod
    def bias(self) -> InputOutputActivationVector | None:
        """Bias.

        May be a reference to a bias parameter in the parent module, if using e.g. a tied bias.
        """
        raise NotImplementedError

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
        raise NotImplementedError


class AbstractAutoencoderModules(TypedDict):
    """Modules of the autoencoder.

    Must include :attr:`encoder` and :attr:`decoder` modules, and may include
    :attr:`pre_encoder_bias` and :attr:`post_decoder_bias` modules.
    """

    encoder: AbstractEncoder
    decoder: AbstractDecoder
    pre_encoder_bias: AbstractOuterBias | None
    post_decoder_bias: AbstractOuterBias | None


class AbstractAutoencoder(Module, ABC):
    """Abstract Sparse Autoencoder Model."""

    device: torch.device | None
    """Device to run the model on."""

    dtype: torch.dtype | None
    """Data type to use for the module."""

    modules: AbstractAutoencoderModules  # type: ignore[assignment] (narrowing)
    """Modules of the autoencoder."""

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

    @property
    @final
    def encoder(self) -> AbstractEncoder:
        """Encoder Module."""
        return self.modules["encoder"]

    @property
    @final
    def decoder(self) -> AbstractDecoder:
        """Decoder Module."""
        return self.modules["decoder"]

    @property
    @final
    def pre_encoder_bias(self) -> Module | None:
        """Pre-Encoder Bias Module."""
        return self.modules.get("pre_encoder_bias")

    @property
    @final
    def post_decoder_bias(self) -> Module | None:
        """Post-Decoder Bias Module."""
        return self.modules.get("post_decoder_bias")
