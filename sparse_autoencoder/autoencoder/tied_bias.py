"""Tied Biases (Pre-Encoder and Post-Decoder)."""
from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


class PreEncoderBias(Module):
    """Tied Pre-Encoder Bias Layer.

    The tied pre-encoder bias is a learned bias term that is subtracted from the input before
    encoding.

    https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias

    Args:
        bias: Tied bias parameter (initialised in the parent module), used for both the pre-encoder
            and post-encoder bias. The original paper initialised this using the geometric median of
            the dataset.
    """

    bias: Float[Tensor, "input_activations"]

    def __init__(
        self,
        bias: Float[Tensor, "input_activations"],
    ) -> None:
        """Initialize the bias layer."""
        super().__init__()

        self.bias = bias

    def forward(
        self, input: Float[Tensor, "*batch input_activations"]
    ) -> Float[Tensor, "*batch input_activations"]:
        """Forward Pass."""
        return input - self.bias


class PostEncoderBias(Module):
    """Tied Post-Encoder Bias Layer.

    The tied post-encoder bias is a learned bias term that is added to the output of the decoder.

    https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias

    Args:
        bias: Tied bias parameter (initialised in the parent module), used for both the pre-encoder
            and post-encoder bias. The original paper initialised this using the geometric median of
            the dataset.
    """

    bias: Float[Tensor, "input_activations"]

    def __init__(
        self,
        bias: Float[Tensor, "input_activations"],
    ) -> None:
        """Initialize the bias layer."""
        super().__init__()

        self.bias = bias

    def forward(
        self, input: Float[Tensor, "*batch input_activations"]
    ) -> Float[Tensor, "*batch input_activations"]:
        """Forward Pass."""
        return input + self.bias
