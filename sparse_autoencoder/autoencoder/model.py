"""The Sparse Autoencoder Model."""
import torch
from jaxtyping import Float
from torch import Tensor
from torch.nn import Linear, Module, ReLU, Sequential
from torch.nn.parameter import Parameter

from sparse_autoencoder.autoencoder.tied_bias import PostEncoderBias, PreEncoderBias


class SparseAutoencoder(Module):
    """Sparse Autoencoder Model.

    Args:
        n_input_features: Number of input features (e.g. `d_mlp` if training on MLP activations from
            TransformerLens).
        n_learned_features: Number of learned features. The initial paper experimented with 1× to
            256× the number of input features, and primarily used 8x.
        geometric_median_dataset: Estimated geometric median of the dataset.
    """

    geometric_median_dataset: Float[Tensor, "input_activations"]
    """Estimated Geometric Median of the Dataset.
    
    Used for initialising :attr:`tied_bias`.
    """

    tied_bias: Float[Parameter, "input_activations"]
    """Tied Bias Parameter.
    
    The same bias is used pre-encoder and post-decoder.
    """

    n_input_features: int
    """Number of Input Features."""

    n_learned_features: int
    """Number of Learned Features."""

    def __init__(
        self,
        n_input_features: int,
        n_learned_features: int,
        geometric_median_dataset: Float[Tensor, "input_activations"],
    ) -> None:
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

        # Create the network
        self.encoder = Sequential(
            PreEncoderBias(self.tied_bias),
            Linear(n_input_features, n_learned_features),
            ReLU(),
        )

        self.decoder = Sequential(
            Linear(n_learned_features, n_input_features),
            PostEncoderBias(self.tied_bias),
        )

    def forward(
        self, input: Float[Tensor, "batch input_activations"]
    ) -> tuple[
        Float[Tensor, "batch learned_activations"],
        Float[Tensor, "batch input_activations"],
    ]:
        """Forward Pass.

        Args:
            input: Input activations (e.g. activations from an MLP layer in a transformer model).

        Returns:
            Tuple of learned activations and decoded activations.
        """
        learned_activations = self.encoder(input)
        decoded_activations = self.decoder(learned_activations)
        return learned_activations, decoded_activations

    def initialize_tied_parameters(self) -> None:
        """Initialize the tied parameters."""
        # The tied bias is initialised as the geometric median of the dataset
        self.tied_bias.data = self.geometric_median_dataset.clone()

    def reset_parameters(self) -> None:
        """Reset the parameters."""
        self.initialize_tied_parameters()
        for module in self.network:
            if "reset_parameters" in dir(module):
                module.reset_parameters()
