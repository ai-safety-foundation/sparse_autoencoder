"""Abstract pipeline."""
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import final

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
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

    total_training_steps: int = 1

    @final
    def __init__(  # noqa: PLR0913
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
        checkpoint_directory: Path | None = None,
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
        self.checkpoint_directory = checkpoint_directory

        source_dataloader = source_dataset.get_dataloader(source_data_batch_size)
        self.source_data = self.stateful_dataloader_iterable(source_dataloader)

    @abstractmethod
    def generate_activations(self, store_size: int) -> TensorActivationStore:
        """Generate activations.

        Args:
            store_size: Number of activations to generate.

        Returns:
            Activation store for the train section.
        """

    @abstractmethod
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: int
    ) -> NeuronActivity:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Number of times each neuron fired.
        """

    @final
    def resample_neurons(
        self,
        neuron_activity: NeuronActivity,
        activation_store: TensorActivationStore,
        train_batch_size: int,
    ) -> None:
        """Resample dead neurons.

        Args:
            neuron_activity: Number of times each neuron fired.
            activation_store: Activation store.
            train_batch_size: Train batch size (also used for resampling).
        """
        if self.activation_resampler is not None:
            # Get the updates
            parameter_updates = self.activation_resampler.resample_dead_neurons(
                neuron_activity=neuron_activity,
                activation_store=activation_store,
                autoencoder=self.autoencoder,
                loss_fn=self.loss,
                train_batch_size=train_batch_size,
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

    @abstractmethod
    def validate_sae(self) -> None:
        """Get validation metrics."""

    @final
    def save_checkpoint(self) -> None:
        """Save the model as a checkpoint."""
        if self.checkpoint_directory:
            file_path: Path = (
                self.checkpoint_directory / f"sae_state_dict-{self.total_training_steps}.pt"
            )
            torch.save(self.autoencoder.state_dict(), file_path)

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
        """Run the full training pipeline.

        Args:
            train_batch_size: Train batch size.
            max_store_size: Maximum size of the activation store.
            max_activations: Maximum total number of activations to train on (the original paper
                used 8bn, although others have had success with 100m+).
            resample_frequency: Frequency at which to resample dead neurons (the original paper used
                every 200m).
            validate_frequency: Frequency at which to get validation metrics.
            checkpoint_frequency: Frequency at which to save a checkpoint.
        """
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

                # Update the counters
                last_resampled += len(activation_store)
                last_validated += len(activation_store)
                last_checkpoint += len(activation_store)

                # Resample dead neurons (if needed)
                progress_bar.set_postfix({"stage": "resample"})
                if last_resampled > resample_frequency and self.activation_resampler is not None:
                    self.resample_neurons(
                        neuron_activity=neuron_activity,
                        activation_store=activation_store,
                        train_batch_size=train_batch_size,
                    )

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

            >>> iterator = AbstractPipeline.stateful_dataloader_iterable(data)
            >>> next(iterator)["int"], next(iterator)["int"]
            (tensor([0]), tensor([1]))

        Args:
            dataloader: PyTorch DataLoader.

        Returns:
            Stateful iterable over the data in the dataloader.

        Yields:
            Data from the dataloader.
        """
        yield from dataloader
