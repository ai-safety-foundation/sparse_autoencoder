"""Abstract pipeline."""
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
    ParameterUpdateResults,
)
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.metrics import (
    AbstractGenerateMetric,
    AbstractTrainMetric,
    AbstractValidationMetric,
)
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TorchTokenizedPrompts
from sparse_autoencoder.tensor_types import (
    NeuronActivity,
)


class AbstractPipeline(ABC):
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    generate_metrics: list[AbstractGenerateMetric]

    train_metrics: list[AbstractTrainMetric]

    validation_metrics: list[AbstractValidationMetric]

    source_model: HookedTransformer

    source_dataset: SourceDataset

    source_data: Iterable[TorchTokenizedPrompts]

    autoencoder: SparseAutoencoder

    loss: AbstractLoss

    cache_name: str

    layer: int

    optimizer: AbstractOptimizerWithReset

    activation_resampler: AbstractActivationResampler | None

    progress_bar: tqdm | None

    @final
    def __init__(
        self,
        cache_name: str,
        layer: int,
        source_model: HookedTransformer,
        autoencoder: SparseAutoencoder,
        source_dataset: SourceDataset,
        optimizer: AbstractOptimizerWithReset,
        loss: AbstractLoss,
        activation_resampler: AbstractActivationResampler | None,
        generate_metrics: list[AbstractGenerateMetric] | None = None,
        train_metrics: list[AbstractTrainMetric] | None = None,
        validation_metrics: list[AbstractValidationMetric] | None = None,
        source_data_batch_size: int = 12,
    ):
        """Initialize the pipeline."""
        self.cache_name = cache_name
        self.layer = layer
        self.generate_metrics = generate_metrics if generate_metrics else []
        self.train_metrics = train_metrics if train_metrics else []
        self.validation_metrics = validation_metrics if validation_metrics else []
        self.source_model = source_model
        self.source_dataset = source_dataset
        self.autoencoder = autoencoder
        self.activation_resampler = activation_resampler
        self.optimizer = optimizer
        self.loss = loss
        self.source_data_batch_size = source_data_batch_size

        source_dataloader = source_dataset.get_dataloader(source_data_batch_size)
        self.source_data = self.stateful_dataloader_iterable(source_dataloader)

    @abstractmethod
    def generate_activations(self, store_size: int) -> TensorActivationStore:
        """Generate activations."""
        raise NotImplementedError

    @abstractmethod
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: int
    ) -> NeuronActivity:
        """Train the sparse autoencoder."""
        raise NotImplementedError

    @abstractmethod
    def resample_neurons(
        self, neuron_activity: NeuronActivity, activation_store: TensorActivationStore
    ) -> ParameterUpdateResults:
        """Resample dead neurons."""
        raise NotImplementedError

    @abstractmethod
    def validate_sae(self) -> None:
        """Get validation metrics."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self) -> None:
        """Save the model as a checkpoint."""
        raise NotImplementedError

    @final
    def run_pipeline(
        self,
        train_batch_size: int,
        max_store_size: int,
        max_activations: int,
        resample_frequency: int,
        validate_frequency: int | None = None,
        checkpoint_frequency: int | None = None,
    ) -> None:
        """Run the full training pipeline."""
        last_resampled: int = 0
        last_validated: int = 0
        last_checkpoint: int = 0
        neuron_activity: NeuronActivity | None = None

        # Get the store size
        store_size: int = (
            max_store_size
            - max_store_size % self.source_data_batch_size * self.source_dataset.context_size
        )

        with tqdm(
            desc="Activations trained on",
            total=max_activations,
        ) as progress_bar:
            for _ in range(0, max_activations, store_size):
                # Generate
                progress_bar.set_postfix({"stage": "generate"})
                activation_store: TensorActivationStore = self.generate_activations(store_size)

                # Train
                progress_bar.set_postfix({"stage": "train"})
                batch_neuron_activity: NeuronActivity = self.train_autoencoder(
                    activation_store, train_batch_size=train_batch_size
                )
                detached_neuron_activity = batch_neuron_activity.detach().cpu()
                if neuron_activity is not None:
                    neuron_activity.add_(detached_neuron_activity)
                else:
                    neuron_activity = detached_neuron_activity

                # Resample dead neurons (if needed)
                progress_bar.set_postfix({"stage": "resample"})
                if last_resampled > resample_frequency and self.activation_resampler is not None:
                    # Get the updates
                    parameter_updates = self.resample_neurons(
                        neuron_activity, activation_store=activation_store
                    )

                    # Update the weights and biases
                    self.autoencoder.encoder.update_dictionary_vectors(
                        parameter_updates.dead_neuron_indices,
                        parameter_updates.dead_encoder_weight_updates,
                    )
                    self.autoencoder.encoder.update_bias(
                        parameter_updates.dead_neuron_indices,
                        parameter_updates.dead_encoder_bias_updates,
                    )
                    self.autoencoder.decoder.update_dictionary_vectors(
                        parameter_updates.dead_neuron_indices,
                        parameter_updates.dead_decoder_weight_updates,
                    )

                    # Reset the optimizer (TODO: Consider resetting just the relevant parameters)
                    self.optimizer.reset_state_all_parameters()

                    # Reset
                    self.last_resampled = 0
                    neuron_activity.zero_()

                # Get validation metrics (if needed)
                progress_bar.set_postfix({"stage": "validate"})
                if validate_frequency is not None and last_validated > validate_frequency:
                    self.validate_sae()
                    self.last_validated = 0

                # Checkpoint (if needed)
                progress_bar.set_postfix({"stage": "checkpoint"})
                if checkpoint_frequency is not None and last_checkpoint > checkpoint_frequency:
                    self.last_checkpoint = 0
                    self.save_checkpoint()

                # Update the progress bar
                progress_bar.update(store_size)

    @staticmethod
    def stateful_dataloader_iterable(
        dataloader: DataLoader[TorchTokenizedPrompts]
    ) -> Iterable[TorchTokenizedPrompts]:
        """Create a stateful dataloader iterable.

        Create an iterable that maintains it's position in the dataloader between loops.

        Examples:
            Without this, when iterating over a DataLoader with 2 loops, each loop get the same data
            (assuming shuffle is turned off). That is to say, the second loop won't maintain the
            position from where the first loop left off.

            >>> from datasets import Dataset
            >>> from torch.utils.data import DataLoader
            >>> def gen():
            ...     yield {"int": 0}
            ...     yield {"int": 1}
            >>> data = DataLoader(Dataset.from_generator(gen))
            >>> next(iter(data))["int"], next(iter(data))["int"]
            (tensor([0]), tensor([0]))

            By contrast if you create a stateful iterable from the dataloader, each loop will get
            different data.

            >>> iterator = stateful_dataloader_iterable(data)
            >>> next(iterator)["int"], next(iterator)["int"]
            (tensor([0]), tensor([1]))

        Args:
            dataloader: PyTorch DataLoader.

        Returns:
            Stateful iterable over the data in the dataloader.
        """
        yield from dataloader
