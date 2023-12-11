"""Abstract activation resampler."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Float, Int, Int64
from torch import Tensor

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.tensor_types import Axis


@dataclass
class ParameterUpdateResults:
    """Parameter update results from resampling dead neurons."""

    dead_neuron_indices: Int64[Tensor, Axis.LEARNT_FEATURE_IDX]
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
    """Abstract activation resampler.

    Developer guide:

        This is just an interface (there are no implemented methods, so it is just a glorified type
        signature).

        It is setup this way so that users can add their own alternative activation resampler with
        the same interface, and then easily drop them in as a replacement for the training pipeline.
        If you find you need it to be more flexible (e.g. have more inputs), please do just submit
        an issue and/or a PR.

        If you want to implement your own activation resampler (i.e. you want to experiment with
        changing the way the weight updates are calculated), you should probably create your own
        custom resampler by extending this class. By default this should be done by installing the
        sparse_autoencoder library (and then extending this class in your own codebase), but if you
        think it would be useful for others as well please do also submit an issue/PR for this.
    """

    @abstractmethod
    def step_resampler(
        self,
        batch_neuron_activity: Int[Tensor, Axis.LEARNT_FEATURE],
        activation_store: TensorActivationStore,
        autoencoder: SparseAutoencoder,
        loss_fn: AbstractLoss,
        train_batch_size: int,
    ) -> ParameterUpdateResults | None:
        """Resample dead neurons.

        Args:
            batch_neuron_activity: Number of times each neuron fired in current batch.
            activation_store: Activation store.
            autoencoder: Sparse autoencoder model.
            loss_fn: Loss function.
            train_batch_size: Train batch size (also used for resampling).

        Returns:
            Indices of dead neurons, and the updates for the encoder and decoder weights and biases.
        """
