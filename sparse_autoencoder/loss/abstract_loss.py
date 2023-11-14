"""Abstract loss."""


from abc import abstractmethod

from sparse_autoencoder.tensor_types import (
    DecodedActivationBatch,
    ItemTensor,
    LearnedActivationBatch,
)


class AbstractLoss:
    """Abstract loss."""

    @abstractmethod
    def forward(
        self,
        learned_activations: LearnedActivationBatch,
        decoded_activations: DecodedActivationBatch,
    ) -> ItemTensor:
        """Calculate loss."""
        raise NotImplementedError
