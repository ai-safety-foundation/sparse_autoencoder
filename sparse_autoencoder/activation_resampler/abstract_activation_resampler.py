"""Abstract activation resampler."""

from abc import ABC, abstractmethod

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.tensor_types import (
    DeadDecoderNeuronWeightUpdates,
    DeadEncoderNeuronBiasUpdates,
    DeadEncoderNeuronWeightUpdates,
    NeuronActivity,
)


class AbstractActivationResampler(ABC):
    """Abstract activation resampler."""

    @abstractmethod
    def resample_dead_neurons(
        self,
        neuron_activity: NeuronActivity,
        store: TensorActivationStore,
        num_input_activations: int = 819_200,
    ) -> tuple[
        DeadEncoderNeuronWeightUpdates, DeadEncoderNeuronBiasUpdates, DeadDecoderNeuronWeightUpdates
    ]:
        """Resample dead neurons.

        Over the course of training, a subset of autoencoder neurons will have zero activity across
        a large number of datapoints. The authors of *Towards Monosemanticity: Decomposing Language
        Models With Dictionary Learning* found that “resampling” these dead neurons during training
        improves the number of likely-interpretable features (i.e., those in the high density
        cluster) and reduces total loss. This resampling may be compatible with the Lottery Ticket
        Hypothesis and increase the number of chances the network has to find promising feature
        directions.

        Warning:
            The optimizer should be reset after applying this function, as the Adam state will be
            incorrect for the modified weights and biases.

        Args:
            neuron_activity: Number of times each neuron fired. store: Activation store.
            store: TODO change.
            num_input_activations: Number of input activations to use when resampling. Will be
                rounded down to be divisible by the batch size, and cannot be larger than the number
                of items currently in the store.
        """
        raise NotImplementedError
