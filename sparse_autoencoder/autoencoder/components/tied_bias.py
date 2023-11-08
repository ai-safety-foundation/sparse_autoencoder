"""Tied Biases (Pre-Encoder and Post-Decoder)."""
from enum import Enum

from jaxtyping import Float
from torch import Tensor
from torch.nn import Module


class TiedBiasPosition(str, Enum):
    """Tied Bias Position."""

    PRE_ENCODER = "pre_encoder"
    POST_DECODER = "post_decoder"


class TiedBias(Module):
    """Tied Bias Layer.

    The tied pre-encoder bias is a learned bias term that is subtracted from the input before
    encoding, and added back after decoding.

    The bias parameter must be initialised in the parent module, and then passed to this layer.

    https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-bias
    """

    _bias_reference: Float[Tensor, " input_activations"]

    _bias_position: TiedBiasPosition

    def __init__(
        self,
        bias: Float[Tensor, " input_activations"],
        position: TiedBiasPosition,
    ) -> None:
        """Initialize the bias layer.

        Args:
            bias: Tied bias parameter (initialised in the parent module), used for both the
                pre-encoder and post-encoder bias. The original paper initialised this using the
                geometric median of the dataset.
            position: Whether this is the pre-encoder or post-encoder bias.
        """
        super().__init__()

        self._bias_reference = bias

        # Support string literals as well as enums
        self._bias_position = position

    def forward(
        self,
        x: Float[Tensor, "*batch input_activations"],
    ) -> Float[Tensor, "*batch input_activations"]:
        """Forward Pass.

        Args:
            x: Input tensor.

        Returns:
            Output of the forward pass.
        """
        # If this is the pre-encoder bias, we subtract the bias from the input.
        if self._bias_position == TiedBiasPosition.PRE_ENCODER:
            return x - self._bias_reference

        # If it's the post-encoder bias, we add the bias to the input.
        return x + self._bias_reference

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return f"position={self._bias_position.value}"
