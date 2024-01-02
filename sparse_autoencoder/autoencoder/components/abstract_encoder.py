"""Abstract Encoder."""
from abc import ABC, abstractmethod
from typing import final

from jaxtyping import Float, Int64
from pydantic import PositiveInt, validate_call
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparse_autoencoder.autoencoder.types import ResetOptimizerParameterDetails
from sparse_autoencoder.tensor_types import Axis


class AbstractEncoder(Module, ABC):
    """Abstract encoder module.

    Typically includes :attr:`weights` and :attr:`bias` parameters, as well as an activation
    function.
    """

    _learnt_features: int
    """Number of learnt features (inputs to this layer)."""

    _input_features: int
    """Number of input features from the source model."""

    _n_components: int | None

    @validate_call
    def __init__(
        self,
        input_features: PositiveInt,
        learnt_features: PositiveInt,
        n_components: PositiveInt | None,
    ) -> None:
        """Initialise the encoder.

        Args:
            input_features: Number of input features to the autoencoder.
            learnt_features: Number of learnt features in the autoencoder.
            n_components: Number of source model components the SAE is trained on.
        """
        super().__init__()
        self._learnt_features = learnt_features
        self._input_features = input_features
        self._n_components = n_components

    @property
    @abstractmethod
    def weight(
        self,
    ) -> Float[
        Parameter,
        Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE),
    ]:
        """Weight.

        Each row in the weights matrix (for a specific component) acts as a dictionary vector,
        representing a single basis element in the learned activation space.
        """

    @property
    @abstractmethod
    def bias(self) -> Float[Parameter, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]:
        """Bias."""

    @property
    @abstractmethod
    def reset_optimizer_parameter_details(self) -> list[ResetOptimizerParameterDetails]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """

    @abstractmethod
    def forward(
        self,
        x: Float[
            Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE)]:
        """Forward pass.

        Args:
            x: Input activations.

        Returns:
            Resulting activations.
        """

    @final
    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[
            Tensor, Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX)
        ],
        updated_dictionary_weights: Float[
            Tensor,
            Axis.names(Axis.COMPONENT_OPTIONAL, Axis.LEARNT_FEATURE_IDX, Axis.INPUT_OUTPUT_FEATURE),
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
                self.weight[
                    component_idx, dictionary_vector_indices[component_idx]
                ] = updated_dictionary_weights[component_idx]

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
