"""Linear encoder layer."""
import math
from typing import final

import torch
from torch.nn import Parameter, ReLU, functional, init

from sparse_autoencoder.autoencoder.components.abstract_encoder import AbstractEncoder
from sparse_autoencoder.tensor_types import (
    EncoderWeights,
    InputOutputActivationBatch,
    LearnedActivationBatch,
    LearntActivationVector,
)


@final
class LinearEncoder(AbstractEncoder):
    r"""Linear encoder layer.

    Linear encoder layer (essentially `nn.Linear`, with a ReLU activation function). Designed to be
    used as the encoder in a sparse autoencoder (excluding any outer tied bias).

    $$
    \begin{align*}
        m &= \text{learned features dimension} \\
        n &= \text{input and output dimension} \\
        b &= \text{batch items dimension} \\
        \overline{\mathbf{x}} \in \mathbb{R}^{b \times n} &= \text{input after tied bias} \\
        W_e \in \mathbb{R}^{m \times n} &= \text{weight matrix} \\
        b_e \in \mathbb{R}^{m} &= \text{bias vector} \\
        f &= \text{ReLU}(\overline{\mathbf{x}} W_e^T + b_e) = \text{LinearEncoder output}
    \end{align*}
    $$
    """

    _learnt_features: int
    """Number of learnt features (inputs to this layer)."""

    _input_features: int
    """Number of decoded features (outputs from this layer)."""

    _weight: EncoderWeights
    """Weight parameter internal state."""

    _bias: LearntActivationVector
    """Bias parameter internal state."""

    @property
    def weight(self) -> EncoderWeights:
        """Weight parameter.

        Each row in the weights matrix acts as a dictionary vector, representing a single basis
        element in the learned activation space.
        """
        return self._weight

    @property
    def bias(self) -> LearntActivationVector:
        """Bias parameter."""
        return self._bias

    activation_function: ReLU
    """Activation function."""

    def __init__(
        self,
        input_features: int,
        learnt_features: int,
    ):
        """Initialize the linear encoder layer."""
        super().__init__()
        self._learnt_features = learnt_features
        self._input_features = input_features

        self._weight = Parameter(
            torch.empty(
                (learnt_features, input_features),
            )
        )
        self._bias = Parameter(torch.zeros(learnt_features))
        self.activation_function = ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize or reset the parameters."""
        # Assumes we are using ReLU activation function (for e.g. leaky ReLU, the `a` parameter and
        # `nonlinerity` must be changed.
        init.kaiming_uniform_(self._weight, nonlinearity="relu")

        # Bias (approach from nn.Linear)
        fan_in = self._weight.size(1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self._bias, -bound, bound)

    def forward(self, x: InputOutputActivationBatch) -> LearnedActivationBatch:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output of the forward pass.
        """
        z = functional.linear(x, self.weight, self.bias)
        return self.activation_function(z)

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return f"in_features={self._input_features}, out_features={self._learnt_features}"
