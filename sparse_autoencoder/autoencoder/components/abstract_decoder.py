"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod
from typing import final

from jaxtyping import Float, Int64
import torch
from torch import Tensor
from torch.nn import Module, Parameter

from sparse_autoencoder.tensor_types import Axis


class AbstractDecoder(Module, ABC):
    """Abstract Decoder Module.

    Typically includes just a :attr:`weight` parameter.
    """

    @property
    @abstractmethod
    def weight(
        self,
    ) -> Float[Parameter, Axis.names(Axis.INPUT_OUTPUT_FEATURE, Axis.LEARNT_FEATURE)]:
        """Weight.

        Each column in the weights matrix acts as a dictionary vector, representing a single basis
        element in the learned activation space.
        """

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
        self,
        x: Float[Tensor, Axis.names(Axis.BATCH, Axis.LEARNT_FEATURE)],
    ) -> Float[Tensor, Axis.names(Axis.BATCH, Axis.INPUT_OUTPUT_FEATURE)]:
        """Forward Pass.

        Args:
            x: Learned activations.

        Returns:
            Decoded activations.
        """

    @abstractmethod
    def reset_parameters(self) -> None:
        """Reset the parameters."""

    @final
    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: Int64[Tensor, Axis.LEARNT_FEATURE_IDX],
        updated_weights: Float[Tensor, Axis.names(Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)],
    ) -> None:
        """Update decoder dictionary vectors.

        Updates the dictionary vectors (rows in the weight matrix) with the given values. Typically
        this is used when resampling neurons (dictionary vectors) that have died.

        Args:
            dictionary_vector_indices: Indices of the dictionary vectors to update.
            updated_weights: Updated weights for just these dictionary vectors.
        """
        if len(dictionary_vector_indices) == 0:
            return

        with torch.no_grad():
            self.weight[:, dictionary_vector_indices] = updated_weights

    @abstractmethod
    def constrain_weights_unit_norm(self) -> None:
        """Constrain the weights to have unit norm."""
