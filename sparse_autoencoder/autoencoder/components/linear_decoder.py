"""Linear decoder layer."""
import math
from typing import final

import einops
from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torch.nn import Module, Parameter, init

from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.utils.tensor_shape import shape_with_optional_dimensions


@final
class LinearDecoder(Module):
    r"""Constrained unit norm linear decoder layer.

    Linear layer decoder, where the dictionary vectors (columns of the weight matrix) are NOT
    constrained to have unit norm. 

    $$ \begin{align*}
        m &= \text{learned features dimension} \\
        n &= \text{input and output dimension} \\
        b &= \text{batch items dimension} \\
        f \in \mathbb{R}^{b \times m} &= \text{encoder output} \\
        W_d \in \mathbb{R}^{n \times m} &= \text{weight matrix} \\
        z \in \mathbb{R}^{b \times m} &= f W_d^T = \text{UnitNormDecoder output (pre-tied bias)}
    \end{align*} $$

    Motivation:
        TODO
    """

    _learnt_features: int
    """Number of learnt features (inputs to this layer)."""

    _decoded_features: int
    """Number of decoded features (outputs from this layer)."""

    _n_components: int | None

    weight: Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE),
    ]
    """Weight parameter.

    Each column in the weights matrix acts as a dictionary vector, representing a single basis
    element in the learned activation space.
    """

    @property
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """
        return [ResetOptimizerParameterDetails(parameter=self.weight, axis=-1)]

    @validate_call
    def __init__(
        self,
        learnt_features: PositiveInt,
        decoded_features: PositiveInt,
        n_components: PositiveInt | None,
    ) -> None:
        """Initialize the constrained unit norm linear layer.

        Args:
            learnt_features: Number of learnt features in the autoencoder.
            decoded_features: Number of decoded (output) features in the autoencoder.
            n_components: Number of source model components the SAE is trained on.
        """
        super().__init__()

        self._learnt_features = learnt_features
        self._decoded_features = decoded_features
        self._n_components = n_components

        # Create the linear layer as per the standard PyTorch linear layer
        self.weight = Parameter(
            torch.empty(
                shape_with_optional_dimensions(n_components, decoded_features, learnt_features),
            )
        )
        self.reset_parameters()

    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)
        ],
        updated_weights: Float[
            Tensor,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE_IDX),
        ],
        component_idx: int | None = None,
    ) -> None:
        """Update decoder dictionary vectors.

        Updates the dictionary vectors (rows in the weight matrix) with the given values. Typically
        this is used when resampling neurons (dictionary vectors) that have died.

        Args:
            dictionary_vector_indices: Indices of the dictionary vectors to update.
            updated_weights: Updated weights for just these dictionary vectors.
            component_idx: Component index to update.

        Raises:
            ValueError: If `component_idx` is not specified when `n_components` is not None.
        """
        if dictionary_vector_indices.numel() == 0:
            return

        with torch.no_grad():
            if component_idx is None:
                if self._n_components is not None:
                    error_message = "component_idx must be specified when n_components is not None"
                    raise ValueError(error_message)

                self.weight[:, dictionary_vector_indices] = updated_weights
            else:
                self.weight[component_idx, :, dictionary_vector_indices] = updated_weights

    def reset_parameters(self) -> None:
        """Initialize or reset the parameters."""
        # Assumes we are using ReLU activation function (for e.g. leaky ReLU, the `a` parameter and
        # `nonlinerity` must be changed.
        init.kaiming_uniform_(self.weight, nonlinearity="relu")
        
    def forward(
        self, x: Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)]:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output of the forward pass.
        """
        return einops.einsum(
            x,
            self.weight,
            f"{Axis.BATCH} ... {Axis.LEARNT_FEATURE}, \
            ... {Axis.INPUT_OUTPUT_FEATURE} {Axis.LEARNT_FEATURE} \
                -> {Axis.BATCH} ... {Axis.INPUT_OUTPUT_FEATURE}",
        )

    def extra_repr(self) -> str:
        """String extra representation of the module."""
        return (
            f"learnt_features={self._learnt_features}, "
            f"decoded_features={self._decoded_features}, "
            f"n_components={self._n_components}"
        )
