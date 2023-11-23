"""Abstract activation resampler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import (
    DeadDecoderNeuronWeightUpdates,
    DeadEncoderNeuronBiasUpdates,
    DeadEncoderNeuronWeightUpdates,
    LearntNeuronIndices,
    NeuronActivity,
)


@dataclass
class ParameterUpdateResults:
    """Parameter update results from resampling dead neurons."""

    dead_neuron_indices: LearntNeuronIndices
    """Dead neuron indices."""

    dead_encoder_weight_updates: DeadEncoderNeuronWeightUpdates
    """Dead encoder weight updates."""

    dead_encoder_bias_updates: DeadEncoderNeuronBiasUpdates
    """Dead encoder bias updates."""

    dead_decoder_weight_updates: DeadDecoderNeuronWeightUpdates
    """Dead decoder weight updates."""


class AbstractActivationResampler(ABC):
    """Abstract activation resampler."""

    _resample_dataset_size: int | None = None
    """Resample dataset size.

    If none, will use the train dataset size.
    """

    @final
    def __init__(self, resample_dataset_size: int | None = None) -> None:
        """Initialize the abstract activation resampler.

        Args:
            resample_dataset_size: Resample dataset size. If none, will use the train dataset size.
        """
        self._resample_dataset_size = resample_dataset_size

    @abstractmethod
    def resample_dead_neurons(
        self,
        neuron_activity: NeuronActivity,
        activation_store: TensorActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> ParameterUpdateResults:
        """Resample dead neurons.

        Args:
            neuron_activity: Number of times each neuron fired.
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            Indices of dead neurons, and the updates for the encoder and decoder weights and biases.
        """
