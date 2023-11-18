"""Linear encoder layer."""
import math
from typing import final

import einops
import torch
from torch.nn import Parameter, ReLU, init

from sparse_autoencoder.autoencoder.components.abstract_encoder import AbstractEncoder
from sparse_autoencoder.tensor_types import (
    EncoderWeights,
    InputOutputActivationBatch,
    LearnedActivationBatch,
    LearntActivationVector,
)


@final
class LinearEncoder(AbstractEncoder):
    """Linear encoder layer."""

    _learnt_features: int
    """Number of learnt features (inputs to this layer)."""

    _input_features: int
    """Number of decoded features (outputs from this layer)."""

    _weight: EncoderWeights

    _bias: LearntActivationVector

    @property
    def weight(self) -> EncoderWeights:
        """Weight."""
        return self._weight

    @property
    def bias(self) -> LearntActivationVector:
        """Bias."""
        return self._bias

    activation_function: ReLU

    def __init__(
        self,
        input_features: int,
        learnt_features: int,
    ):
        """Initialize the linear encoder layer."""
        super().__init__()
        self._learnt_features = learnt_features
        self._input_features = input_features
        self.activation_function = ReLU()

        self._weight = Parameter(
            torch.empty(
                (learnt_features, input_features),
            )
        )

        self._bias = Parameter(torch.zeros(learnt_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize or reset the parameters."""
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self._weight, a=math.sqrt(5))

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
        learned_activation_batch: LearnedActivationBatch = einops.einsum(
            x,
            self.weight,
            "batch input_output_feature, \
                learnt_feature_dim input_output_feature_dim \
                -> batch learnt_feature_dim",
        )

        learned_activation_batch = einops.einsum(
            learned_activation_batch,
            self.bias,
            "batch learnt_feature_dim, \
                learnt_feature_dim -> batch learnt_feature_dim",
        )

        return self.activation_function(learned_activation_batch)

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return f"in_features={self._input_features}, out_features={self._learnt_features}"
