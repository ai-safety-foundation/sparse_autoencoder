"""The Sparse Autoencoder Model."""

from typing import final

from jaxtyping import Float
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from sparse_autoencoder.autoencoder.abstract_autoencoder import (
    AbstractAutoencoder,
    AutoencoderForwardPassResult,
)
from sparse_autoencoder.autoencoder.components.linear_encoder import LinearEncoder
from sparse_autoencoder.autoencoder.components.tied_bias import TiedBias, TiedBiasPosition
from sparse_autoencoder.autoencoder.components.unit_norm_decoder import UnitNormDecoder
from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


@final
class SparseAutoencoder(AbstractAutoencoder):
    """Sparse Autoencoder Model."""

    geometric_median_dataset: Float[
        Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Estimated Geometric Median of the Dataset.

    Used for initialising :attr:`tied_bias`.
    """

    tied_bias: Float[
        Parameter, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Tied Bias Parameter.

    The same bias is used pre-encoder and post-decoder.
    """

    n_components: int | None
    """Number of source model components the SAE is trained on."""

    n_input_features: int
    """Number of Input Features."""

    n_learned_features: int
    """Number of Learned Features."""

    pre_encoder_bias: TiedBias
    """Pre-Encoder Bias."""

    encoder: LinearEncoder
    """Encoder."""

    decoder: UnitNormDecoder
    """Decoder."""

    post_decoder_bias: TiedBias
    """Post-Decoder Bias."""

    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        n_input_features: PositiveInt,
        n_learned_features: PositiveInt,
        geometric_median_dataset: Float[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ]
        | None = None,
        n_components: PositiveInt | None = None,
    ) -> None:
        """Initialize the Sparse Autoencoder Model.

        Args:
            n_input_features: Number of input features (e.g. `d_mlp` if training on MLP activations
                from TransformerLens).
            n_learned_features: Number of learned features. The initial paper experimented with 1 to
                256 times the number of input features, and primarily used a multiple of 8.
            geometric_median_dataset: Estimated geometric median of the dataset.
            n_components: Number of source model components the SAE is trained on. This is useful if
                you want to train the SAE on several components of the source model at once. If
                `None`, the SAE is assumed to be trained on just one component (in this case the
                model won't contain a component axis in any of the parameters).
        """
        super().__init__()

        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features
        self.n_components = n_components

        # Store the geometric median of the dataset (so that we can reset parameters). This is not a
        # parameter itself (the tied bias parameter is used for that), so gradients are disabled.
        tied_bias_shape = shape_with_optional_dimensions(n_components, n_input_features)
        if geometric_median_dataset is not None:
            self.geometric_median_dataset = geometric_median_dataset.clone()
            self.geometric_median_dataset.requires_grad = False
        else:
            self.geometric_median_dataset = torch.zeros(tied_bias_shape)
            self.geometric_median_dataset.requires_grad = False

        # Initialize the tied bias
        self.tied_bias = Parameter(torch.empty(tied_bias_shape))
        self.initialize_tied_parameters()

        # Initialize the components
        self.pre_encoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.PRE_ENCODER)

        self.encoder = LinearEncoder(
            input_features=n_input_features,
            learnt_features=n_learned_features,
            n_components=n_components,
        )

        self.decoder = UnitNormDecoder(
            learnt_features=n_learned_features,
            decoded_features=n_input_features,
            n_components=n_components,
        )

        self.post_decoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.POST_DECODER)

    def forward(
        self,
        x: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> AutoencoderForwardPassResult:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Tuple of learned activations and decoded activations.
        """
        x = self.pre_encoder_bias(x)
        learned_activations = self.encoder(x)
        x = self.decoder(learned_activations)
        decoded_activations = self.post_decoder_bias(x)

        return AutoencoderForwardPassResult(learned_activations, decoded_activations)

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

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
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

    def post_backwards_hook(self) -> None:
        """Hook to be called after each learning step.

        This can be used to e.g. constrain weights to unit norm.
        """
        self.decoder.constrain_weights_unit_norm()
