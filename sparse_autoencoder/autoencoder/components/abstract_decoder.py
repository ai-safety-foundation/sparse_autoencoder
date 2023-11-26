"""Abstract Sparse Autoencoder Model."""
from abc import ABC, abstractmethod
from typing import final

import torch
from torch.nn import Module

from sparse_autoencoder.tensor_types import (
    DeadDecoderNeuronWeightUpdates,
    DecoderWeights,
    InputOutputActivationBatch,
    LearnedActivationBatch,
    LearntNeuronIndices,
)


class AbstractDecoder(Module, ABC):
    """Abstract Decoder Module.

    Typically includes just a :attr:`weight` parameter.
    """

    @property
    @abstractmethod
    def weight(self) -> DecoderWeights:
        """Weight.

        Each column in the weights matrix acts as a dictionary vector, representing a single basis
        element in the learned activation space.
        """

    @abstractmethod
    def forward(
        self,
        x: LearnedActivationBatch,
    ) -> InputOutputActivationBatch:
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
        dictionary_vector_indices: LearntNeuronIndices,
        updated_weights: DeadDecoderNeuronWeightUpdates,
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
