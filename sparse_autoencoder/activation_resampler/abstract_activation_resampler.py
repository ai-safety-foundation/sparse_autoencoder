"""Abstract activation resampler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

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

    @abstractmethod
    def resample_dead_neurons(
        self,
        neuron_activity: NeuronActivity,
        activation_store: TensorActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
        num_inputs: int = 819_200,
    ) -> ParameterUpdateResults:
        """Resample dead neurons.

        Args:
            neuron_activity: Number of times each neuron fired.
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).
            num_inputs: Number of input activations to use when resampling. Will be rounded down to
                be divisible by the batch size, and cannot be larger than the number of items
                currently in the store.
        """
        raise NotImplementedError
