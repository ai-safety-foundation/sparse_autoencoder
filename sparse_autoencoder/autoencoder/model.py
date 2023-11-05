"""The Sparse Autoencoder Model."""
from jaxtyping import Float
import torch
from torch import Tensor
from torch.nn import Module, ReLU, Sequential
from torch.nn.parameter import Parameter

from sparse_autoencoder.autoencoder.tied_bias import PostEncoderBias, PreEncoderBias
from sparse_autoencoder.autoencoder.unit_norm_linear import ConstrainedUnitNormLinear


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

    def __init__(
        self,
        n_input_features: int,
        n_learned_features: int,
        geometric_median_dataset: Float[Tensor, " input_activations"],
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

        # Create the network
        self.encoder = Sequential(
            PreEncoderBias(self.tied_bias),
            ConstrainedUnitNormLinear(n_input_features, n_learned_features),
            ReLU(),
        )

        self.decoder = Sequential(
            ConstrainedUnitNormLinear(n_learned_features, n_input_features, bias=False),
            PostEncoderBias(self.tied_bias),
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
        self.tied_bias.data = self.geometric_median_dataset.clone()

    def reset_parameters(self) -> None:
        """Reset the parameters."""
        self.initialize_tied_parameters()
        for module in self.network:
            if "reset_parameters" in dir(module):
                module.reset_parameters()

    def make_decoder_weights_and_grad_unit_norm(self) -> None:
        """Make the decoder weights and gradients unit norm.

        Unit norming the dictionary vectors, which are essentially the columns of the encoding and
        decoding matrices, serves a few purposes:

            1. It helps with numerical stability, by preventing the dictionary vectors from growing
                too large.
            2. It acts as a form of regularization, preventing overfitting by not allowing any one
                feature to dominate the representation. It limits the capacity of the model by
                forcing the dictionary vectors to live on the hypersphere of radius 1.
            3. It encourages sparsity. Since the dictionary vectors have a fixed length, the model
                must carefully select which features to activate in order to best reconstruct the
                input.

        Each input vector is a row of size `(1, n_input_features)`. The encoding matrix is then of
        shape `(n_input_features, n_learned_features)`. The columns are the dictionary vectors, i.e.
        each one projects the input vector onto a basis vector in the learned feature space.

        Each decoding matrix is of shape `(n_learned_features, n_input_features)`, with the output
        vectors as rows of size `(1, n_input_features)`. The columns of the decoding matrix are the
        dictionary vectors that reconstruct the learned features in the input space.

        Note that the *Towards Monosemanticity: Decomposing Language Models With Dictionary
        Learning* paper found that removing the gradient information parallel to the dictionary
        vectors before applying the gradient step, rather than resetting the dictionary vectors to
        unit norm after each gradient step, [results in a small but real reduction in total
        loss](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-optimization).

        Approach:
            The gradient with respect to the decoder weights is of shape `(n_learned_features,
            n_input_features)` (and similarly for the encoder weights it's just the same shape as
            the weights themselves). By subtracting the projection of the gradient onto the
            dictionary vectors, we remove the component of the gradient that is parallel to the
            dictionary vectors and just keep the component that is orthogonal to the dictionary
            vectors (i.e. moving around the hypersphere). The result is that the gradient moves
            around the hypersphere, but never moves towards or away from the center. Note this does
            mean that we must assume

        TODO: Implement this.

        TODO: Consider creating a custom module to do this.
        """
        raise NotImplementedError

    def save_to_hf(self) -> None:
        """Save the model to Hugging Face."""
        raise NotImplementedError

    def load_from_hf(self) -> None:
        """Load the model from Hugging Face."""
        raise NotImplementedError
