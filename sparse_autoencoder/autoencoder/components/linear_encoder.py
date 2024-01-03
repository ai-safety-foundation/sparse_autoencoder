"""Linear encoder layer."""
import math
from typing import final

import einops
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torch.nn import Module, Parameter, ReLU, init

from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


@final
class LinearEncoder(Module):
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
    """Number of input features from the source model."""

    _n_components: int | None

    _weight: Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE),
    ]
    """Weight parameter internal state."""

    _bias: Float[Parameter, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]
    """Bias parameter internal state."""

    @property
    def weight(
        self,
    ) -> Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE),
    ]:
        """Weight parameter.

        Each row in the weights matrix acts as a dictionary vector, representing a single basis
        element in the learned activation space.
        """
        return self._weight

    @property
    def bias(self) -> Float[Parameter, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]:
        """Bias parameter."""
        return self._bias

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """
        return [
            ResetOptimizerParameterDetails(parameter=self.weight, axis=-2),
            ResetOptimizerParameterDetails(parameter=self.bias, axis=-1),
        ]

    activation_function: ReLU
    """Activation function."""

    @validate_call
    def __init__(
        self,
        input_features: PositiveInt,
        learnt_features: PositiveInt,
        n_components: PositiveInt | None,
    ):
        """Initialize the linear encoder layer.

        Args:
            input_features: Number of input features to the autoencoder.
            learnt_features: Number of learnt features in the autoencoder.
            n_components: Number of source model components the SAE is trained on.
        """
        super().__init__()

        self._learnt_features = learnt_features
        self._input_features = input_features
        self._n_components = n_components

        self._weight = Parameter(
            torch.empty(
                shape_with_optional_dimensions(n_components, learnt_features, input_features),
            )
        )
        self._bias = Parameter(
            torch.zeros(shape_with_optional_dimensions(n_components, learnt_features))
        )
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

    def forward(
        self,
        x: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output of the forward pass.
        """
        z = (
            einops.einsum(
                x,
                self.weight,
                f"{Axis.BATCH} ... {Axis.INPUT_OUTPUT_FEATURE}, \
                    ... {Axis.LEARNT_FEATURE} {Axis.INPUT_OUTPUT_FEATURE} \
                    -> {Axis.BATCH} ... {Axis.LEARNT_FEATURE}",
            )
            + self.bias
        )

        return self.activation_function(z)

    @final
    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX)],
        updated_dictionary_weights: Float[
            Tensor, Axis.names(Axis.LEARNT_FEATURE_IDX, Axis.INPUT_OUTPUT_FEATURE)
        ],
        component_idx: int | None = None,
    ) -> None:
        """Update encoder dictionary vectors.

        Updates the dictionary vectors (columns in the weight matrix) with the given values.

        Args:
            dictionary_vector_indices: Indices of the dictionary vectors to update.
            updated_dictionary_weights: Updated weights for just these dictionary vectors.
            component_idx: Component index to update.

        Raises:
            ValueError: If there are multiple components and `component_idx` is not specified.
        """
        if dictionary_vector_indices.numel() == 0:
            return

        with torch.no_grad():
            if component_idx is None:
                if self._n_components is not None:
                    error_message = "component_idx must be specified when n_components is not None"
                    raise ValueError(error_message)

                self.weight[dictionary_vector_indices] = updated_dictionary_weights
            else:
                self.weight[component_idx, dictionary_vector_indices] = updated_dictionary_weights

    @final
    def update_bias(
        self,
        update_parameter_indices: Int64[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)
        ],
        updated_bias_features: Float[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)
        ],
        component_idx: int | None = None,
    ) -> None:
        """Update encoder bias.

        Args:
            update_parameter_indices: Indices of the bias features to update.
            updated_bias_features: Updated bias features for just these indices.
            component_idx: Component index to update.

        Raises:
            ValueError: If there are multiple components and `component_idx` is not specified.
        """
        if update_parameter_indices.numel() == 0:
            return

        with torch.no_grad():
            if component_idx is None:
                if self._n_components is not None:
                    error_message = "component_idx must be specified when n_components is not None"
                    raise ValueError(error_message)

                self.bias[update_parameter_indices] = updated_bias_features
            else:
                self.bias[component_idx, update_parameter_indices] = updated_bias_features

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return (
            f"input_features={self._input_features}, "
            f"learnt_features={self._learnt_features}, "
            f"n_components={self._n_components}"
        )
