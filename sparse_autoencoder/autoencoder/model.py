"""The Sparse Autoencoder Model."""

from typing import final

import torch
from torch.nn.parameter import Parameter

from sparse_autoencoder.autoencoder.abstract_autoencoder import AbstractAutoencoder
from sparse_autoencoder.autoencoder.components.linear_encoder import LinearEncoder
from sparse_autoencoder.autoencoder.components.tied_bias import TiedBias, TiedBiasPosition
from sparse_autoencoder.autoencoder.components.unit_norm_decoder import UnitNormDecoder
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    InputOutputActivationVector,
    LearnedActivationBatch,
)


@final
class SparseAutoencoder(AbstractAutoencoder):
    """Sparse Autoencoder Model."""

    geometric_median_dataset: InputOutputActivationVector
    """Estimated Geometric Median of the Dataset.

    Used for initialising :attr:`tied_bias`.
    """

    tied_bias: InputOutputActivationBatch
    """Tied Bias Parameter.

    The same bias is used pre-encoder and post-decoder.
    """

    n_input_features: int
    """Number of Input Features."""

    n_learned_features: int
    """Number of Learned Features."""

    _pre_encoder_bias: TiedBias

    _encoder: LinearEncoder

    _decoder: UnitNormDecoder

    _post_decoder_bias: TiedBias

    @property
    def pre_encoder_bias(self) -> TiedBias:
        """Pre-encoder bias."""
        return self._pre_encoder_bias

    @property
    def encoder(self) -> LinearEncoder:
        """Encoder."""
        return self._encoder

    @property
    def decoder(self) -> UnitNormDecoder:
        """Decoder."""
        return self._decoder

    @property
    def post_decoder_bias(self) -> TiedBias:
        """Post-decoder bias."""
        return self._post_decoder_bias

    def __init__(
        self,
        n_input_features: int,
        n_learned_features: int,
        geometric_median_dataset: InputOutputActivationVector,
    ) -> None:
        """Initialize the Sparse Autoencoder Model.

        Args:
            n_input_features: Number of input features (e.g. `d_mlp` if training on MLP activations
                from TransformerLens).
            n_learned_features: Number of learned features. The initial paper experimented with 1 to
                256 times the number of input features, and primarily used a multiple of 8.
            geometric_median_dataset: Estimated geometric median of the dataset.
        """
        super().__init__()

        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features

        # Store the geometric median of the dataset (so that we can reset parameters). This is not a
        # parameter itself (the tied bias parameter is used for that), so gradients are disabled.
        self.geometric_median_dataset = geometric_median_dataset.clone()
        self.geometric_median_dataset.requires_grad = False

        # Initialize the tied bias
        self.tied_bias = Parameter(torch.empty(n_input_features))
        self.initialize_tied_parameters()

        self._pre_encoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.PRE_ENCODER)

        self._encoder = LinearEncoder(
            input_features=n_input_features, learnt_features=n_learned_features
        )

        self._decoder = UnitNormDecoder(
            learnt_features=n_learned_features, decoded_features=n_input_features
        )

        self._post_decoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.POST_DECODER)

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
        x = self._pre_encoder_bias(x)
        learned_activations = self._encoder(x)
        x = self._decoder(learned_activations)
        decoded_activations = self._post_decoder_bias(x)
        return learned_activations, decoded_activations

    def initialize_tied_parameters(self) -> None:
        """Initialize the tied parameters."""
        # The tied bias is initialised as the geometric median of the dataset
        self.tied_bias.data = self.geometric_median_dataset

    def reset_parameters(self) -> None:
        """Reset the parameters."""
        self.initialize_tied_parameters()
        for module in self.network:
            if "reset_parameters" in dir(module):
                module.reset_parameters()
