"""Abstract Encoder."""
from abc import ABC, abstractmethod
from typing import final

from jaxtyping import Float, Int64
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparse_autoencoder.tensor_types import Axis


class AbstractEncoder(Module, ABC):
    """Abstract encoder module.

    Typically includes :attr:`weights` and :attr:`bias` parameters, as well as an activation
    function.
    """

    @property
    @abstractmethod
    def weight(
        self,
    ) -> Float[Parameter, Axis.names(Axis.LEARNT_FEATURE, Axis.INPUT_OUTPUT_FEATURE)]:
        """Weight.

        Each row in the weights matrix acts as a dictionary vector, representing a single basis
        element in the learned activation space.
        """

    @property
    @abstractmethod
    def bias(self) -> Float[Parameter, Axis.LEARNT_FEATURE]:
        """Bias."""

    @property
    @abstractmethod
    def reset_optimizer_parameter_details(self) -> list[tuple[Parameter, int]]:
        """Reset optimizer parameter details.

        Details of the parameters that should be reset in the optimizer, when resetting
        dictionary vectors.

        Returns:
            List of tuples of the form `(parameter, axis)`, where `parameter` is the parameter to
            reset (e.g. encoder.weight), and `axis` is the axis of the parameter to reset.
        """

    @abstractmethod
    def forward(
        self, x: Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)]:
        """Forward pass.

        Args:
            x: Input activations.

        Returns:
            Resulting activations.
        """

    @final
    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[Tensor, Axis.LEARNT_FEATURE_IDX],
        updated_dictionary_weights: Float[
            Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
        ],
    ) -> None:
        """Update encoder dictionary vectors.

        Updates the dictionary vectors (columns in the weight matrix) with the given values.

        Args:
            dictionary_vector_indices: Indices of the dictionary vectors to update.
            updated_dictionary_weights: Updated weights for just these dictionary vectors.
        """
        if len(dictionary_vector_indices) == 0:
            return

        with torch.no_grad():
            self.weight[dictionary_vector_indices, :] = updated_dictionary_weights

    @final
    def update_bias(
        self,
        update_parameter_indices: Int64[Tensor, Axis.INPUT_OUTPUT_FEATURE],
        updated_bias_features: Float[Tensor, Axis.LEARNT_FEATURE] | float,
    ) -> None:
        """Update encoder bias.

        Args:
            update_parameter_indices: Indices of the bias features to update.
            updated_bias_features: Updated bias features for just these indices.
        """
        if len(update_parameter_indices) == 0:
            return

        with torch.no_grad():
            self.bias[update_parameter_indices] = updated_bias_features
