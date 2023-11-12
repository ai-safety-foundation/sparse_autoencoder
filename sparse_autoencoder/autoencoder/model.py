"""The Sparse Autoencoder Model."""
from collections import OrderedDict

from jaxtyping import Float
import torch
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential
from torch.nn.parameter import Parameter

from sparse_autoencoder.autoencoder.components.tied_bias import TiedBias, TiedBiasPosition
from sparse_autoencoder.autoencoder.components.unit_norm_linear import ConstrainedUnitNormLinear


class SparseAutoencoder(Module):
    """Sparse Autoencoder Model."""

    geometric_median_dataset: Float[Tensor, " input_activations"]
    """Estimated Geometric Median of the Dataset.

    Used for initialising :attr:`tied_bias`.
    """

    tied_bias: Float[Parameter, " input_activations"]
    """Tied Bias Parameter.

    The same bias is used pre-encoder and post-decoder.
    """

    n_input_features: int
    """Number of Input Features."""

    n_learned_features: int
    """Number of Learned Features."""

    device: torch.device | None
    """Device to run the model on."""

    dtype: torch.dtype | None
    """Data type to use for the model."""

    encoder: Sequential
    """Encoder Module."""

    decoder: Sequential
    """Decoder Module."""

    def __init__(
        self,
        n_input_features: int,
        n_learned_features: int,
        geometric_median_dataset: Float[Tensor, " input_activations"],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the Sparse Autoencoder Model.

        Args:
            n_input_features: Number of input features (e.g. `d_mlp` if training on MLP activations
                from TransformerLens).
            n_learned_features: Number of learned features. The initial paper experimented with 1 to
                256 times the number of input features, and primarily used a multiple of 8.
            geometric_median_dataset: Estimated geometric median of the dataset.
            device: Device to run the model on.
            dtype: Data type to use for the model.
        """
        super().__init__()

        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features
        self.device = device
        self.dtype = dtype

        # Store the geometric median of the dataset (so that we can reset parameters). This is not a
        # parameter itself (the tied bias parameter is used for that), so gradients are disabled.
        self.geometric_median_dataset = geometric_median_dataset.clone()
        self.geometric_median_dataset.requires_grad = False

        # Initialize the tied bias
        self.tied_bias = Parameter(torch.empty((n_input_features), device=device, dtype=dtype))
        self.initialize_tied_parameters()

        self.encoder = Sequential(
            OrderedDict(
                {
                    "TiedBias": TiedBias(self.tied_bias, TiedBiasPosition.PRE_ENCODER),
                    "Linear": Linear(
                        n_input_features, n_learned_features, bias=True, device=device, dtype=dtype
                    ),
                    "ReLU": ReLU(),
                }
            )
        )

        self.decoder = Sequential(
            OrderedDict(
                {
                    "ConstrainedUnitNormLinear": ConstrainedUnitNormLinear(
                        n_learned_features, n_input_features, bias=False, device=device, dtype=dtype
                    ),
                    "TiedBias": TiedBias(self.tied_bias, TiedBiasPosition.POST_DECODER),
                }
            )
        )

    def forward(
        self,
        x: Float[Tensor, "batch input_activations"],
    ) -> tuple[
        Float[Tensor, "batch learned_activations"],
        Float[Tensor, "batch input_activations"],
    ]:
        """Forward Pass.

        Args:
            x: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Tuple of learned activations and decoded activations.
        """
        learned_activations = self.encoder(x)
        decoded_activations = self.decoder(learned_activations)
        return learned_activations, decoded_activations

    def initialize_tied_parameters(self) -> None:
        """Initialize the tied parameters."""
        # The tied bias is initialised as the geometric median of the dataset
        self.tied_bias.data = self.geometric_median_dataset.clone().to(
            device=self.device, dtype=self.dtype
        )

    def reset_parameters(self) -> None:
        """Reset the parameters."""
        self.initialize_tied_parameters()
        for module in self.network:
            if "reset_parameters" in dir(module):
                module.reset_parameters()

    def save_to_hf(self) -> None:
        """Save the model to Hugging Face."""
        raise NotImplementedError

    def load_from_hf(self) -> None:
        """Load the model from Hugging Face."""
        raise NotImplementedError
