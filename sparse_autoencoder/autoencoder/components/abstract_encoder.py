"""Abstract Encoder."""
from abc import ABC, abstractmethod
from typing import final

import torch
from torch.nn import Module

from sparse_autoencoder.tensor_types import (
    DeadEncoderNeuronWeightUpdates,
    EncoderWeights,
    InputOutputActivationBatch,
    InputOutputNeuronIndices,
    LearnedActivationBatch,
    LearntActivationVector,
    LearntNeuronIndices,
)


class AbstractEncoder(Module, ABC):
    """Abstract encoder module.

    Typically includes :attr:`weights` and :attr:`bias` parameters, as well as an activation
    function.
    """

    @property
    @abstractmethod
    def weight(self) -> EncoderWeights:
        """Weight."""

    @property
    @abstractmethod
    def bias(self) -> LearntActivationVector:
        """Bias."""

    @abstractmethod
    def forward(self, x: InputOutputActivationBatch) -> LearnedActivationBatch:
        """Forward pass.

        Args:
            x: Input activations.

        Returns:
            Resulting activations.
        """

    @final
    def update_dictionary_vectors(
        self,
        dictionary_vector_indices: LearntNeuronIndices,
        updated_dictionary_weights: DeadEncoderNeuronWeightUpdates,
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
        update_parameter_indices: InputOutputNeuronIndices,
        updated_bias_features: LearntActivationVector | float,
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
