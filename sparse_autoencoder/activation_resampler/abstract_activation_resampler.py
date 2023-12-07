"""Abstract activation resampler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Float, Int
from torch import Tensor

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import Axis


@dataclass
class ParameterUpdateResults:
    """Parameter update results from resampling dead neurons."""

    dead_neuron_indices: Int[Tensor, Axis.LEARNT_FEATURE_IDX]
    """Dead neuron indices."""

    dead_encoder_weight_updates: Float[
        Tensor, Axis.names(Axis.DEAD_FEATURE, Axis.INPUT_OUTPUT_FEATURE)
    ]
    """Dead encoder weight updates."""

    dead_encoder_bias_updates: Float[Tensor, Axis.DEAD_FEATURE]
    """Dead encoder bias updates."""

    dead_decoder_weight_updates: Float[
        Tensor, Axis.names(Axis.INPUT_OUTPUT_FEATURE, Axis.DEAD_FEATURE)
    ]
    """Dead decoder weight updates."""


class AbstractActivationResampler(ABC):
    """Abstract activation resampler."""

    _resample_dataset_size: int | None = None
    """Resample dataset size.

    If none, will use the train dataset size.
    """

    collated_neuron_activity: Int[Tensor, Axis.LEARNT_FEATURE] | None
    """Collated neuron activity.

    How many times each neuron has fired, over the current collation window.
    """

    @abstractmethod
    def step_resampler(
        self,
        last_resampled: int,
        batch_neuron_activity: Int[Tensor, Axis.LEARNT_FEATURE],
        activation_store: TensorActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> ParameterUpdateResults | None:
        """Resample dead neurons.

        Args:
            last_resampled: Number of steps since last resampled.
            batch_neuron_activity: Number of times each neuron fired in current batch.
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            Indices of dead neurons, and the updates for the encoder and decoder weights and biases.
        """

    @abstractmethod
    def resample_dead_neurons(
        self,
        activation_store: TensorActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> ParameterUpdateResults:
        """Resample dead neurons.

        Args:
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            Indices of dead neurons, and the updates for the encoder and decoder weights and biases.
        """
