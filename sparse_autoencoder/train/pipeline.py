"""Default pipeline."""
from collections.abc import Iterable
from functools import partial
from pathlib import Path
from typing import final

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import wandb

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
)
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss
from sparse_autoencoder.metrics.metrics_container import MetricsContainer, default_metrics
from sparse_autoencoder.metrics.resample.abstract_resample_metric import ResampleMetricData
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.validate.abstract_validate_metric import ValidationMetricData
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TorchTokenizedPrompts
from sparse_autoencoder.source_model.replace_activations_hook import replace_activations_hook
from sparse_autoencoder.source_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.source_model.zero_ablate_hook import zero_ablate_hook
from sparse_autoencoder.tensor_types import BatchTokenizedPrompts, NeuronActivity
from sparse_autoencoder.train.utils import get_model_device


class Pipeline:
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    activation_resampler: AbstractActivationResampler | None
    """Activation resampler to use."""

    autoencoder: SparseAutoencoder
    """Sparse autoencoder to train."""

    cache_name: str
    """Name of the cache to use in the source model (hook point)."""

    layer: int
    """Layer to get activations from with the source model."""

    log_frequency: int
    """Frequency at which to log metrics (in steps)."""

    loss: AbstractLoss
    """Loss function to use."""

    metrics: MetricsContainer
    """Metrics to use."""

    optimizer: AbstractOptimizerWithReset
    """Optimizer to use."""

    progress_bar: tqdm | None
    """Progress bar for the pipeline."""

    source_data: Iterable[TorchTokenizedPrompts]
    """Iterable over the source data."""

    source_dataset: SourceDataset
    """Source dataset to generate activation data from (tokenized prompts)."""

    source_model: HookedTransformer
    """Source model to get activations from."""

    total_training_steps: int = 1
    """Total number of training steps state."""

    @final
    def __init__(  # noqa: PLR0913
        self,
        activation_resampler: AbstractActivationResampler | None,
        autoencoder: SparseAutoencoder,
        cache_name: str,
        layer: int,
        loss: AbstractLoss,
        optimizer: AbstractOptimizerWithReset,
        source_dataset: SourceDataset,
        source_model: HookedTransformer,
        checkpoint_directory: Path | None = None,
        log_frequency: int = 100,
        metrics: MetricsContainer = default_metrics,
        source_data_batch_size: int = 12,
    ) -> None:
        """Initialize the pipeline.

        Args:
            activation_resampler: Activation resampler to use.
            autoencoder: Sparse autoencoder to train.
            cache_name: Name of the cache to use in the source model (hook point).
            layer: Layer to get activations from with the source model.
            loss: Loss function to use.
            optimizer: Optimizer to use.
            source_dataset: Source dataset to get data from.
            source_model: Source model to get activations from.
            checkpoint_directory: Directory to save checkpoints to.
            log_frequency: Frequency at which to log metrics (in steps)
            metrics: Metrics to use.
            source_data_batch_size: Batch size for the source data.
        """
        self.activation_resampler = activation_resampler
        self.autoencoder = autoencoder
        self.cache_name = cache_name
        self.checkpoint_directory = checkpoint_directory
        self.layer = layer
        self.log_frequency = log_frequency
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.source_data_batch_size = source_data_batch_size
        self.source_dataset = source_dataset
        self.source_model = source_model

        source_dataloader = source_dataset.get_dataloader(source_data_batch_size)
        self.source_data = self.stateful_dataloader_iterable(source_dataloader)

    def generate_activations(self, store_size: int) -> TensorActivationStore:
        """Generate activations.

        Args:
            store_size: Number of activations to generate.

        Returns:
            Activation store for the train section.

        Raises:
            ValueError: If the store size is not positive or is not divisible by the batch size.
        """
        # Check the store size is positive and divisible by the batch size
        if store_size <= 0:
            error_message = f"Store size must be positive, got {store_size}"
            raise ValueError(error_message)
        if store_size % self.source_data_batch_size != 0:
            error_message = (
                f"Store size must be divisible by the batch size ({self.source_data_batch_size}), "
                f"got {store_size}"
            )
            raise ValueError(error_message)

        # Setup the store
        num_neurons: int = self.autoencoder.n_input_features
        source_model_device: torch.device = get_model_device(self.source_model)
        store = TensorActivationStore(store_size, num_neurons)

        # Add the hook to the model (will automatically store the activations every time the model
        # runs)
        self.source_model.remove_all_hook_fns()
        hook = partial(store_activations_hook, store=store)
        self.source_model.add_hook(self.cache_name, hook)

        # Loop through the dataloader until the store reaches the desired size
        with torch.no_grad():
            for batch in self.source_data:
                input_ids: BatchTokenizedPrompts = batch["input_ids"].to(source_model_device)
                self.source_model.forward(
                    input_ids, stop_at_layer=self.layer + 1, prepend_bos=False
                )  # type: ignore (TLens is typed incorrectly)

                if len(store) >= store_size:
                    break

        self.source_model.remove_all_hook_fns()
        store.shuffle()

        return store

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
        autoencoder_device: torch.device = get_model_device(self.autoencoder)

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
        )

        learned_activations_fired_count: NeuronActivity = torch.zeros(
            self.autoencoder.n_learned_features, dtype=torch.int32, device=autoencoder_device
        )

        for store_batch in activations_dataloader:
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(autoencoder_device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder(batch)

            # Get loss & metrics
            metrics = {}
            total_loss, loss_metrics = self.loss.batch_scalar_loss_with_log(
                batch, learned_activations, reconstructed_activations
            )
            metrics.update(loss_metrics)

            with torch.no_grad():
                for metric in self.metrics.train_metrics:
                    calculated = metric.calculate(
                        TrainMetricData(batch, learned_activations, reconstructed_activations)
                    )
                    metrics.update(calculated)

            # Store count of how many neurons have fired
            with torch.no_grad():
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))

            # Backwards pass
            total_loss.backward()
            self.optimizer.step()
            self.autoencoder.decoder.constrain_weights_unit_norm()

            # Log
            if wandb.run is not None and self.total_training_steps % self.log_frequency == 0:
                wandb.log(
                    data={**metrics, **loss_metrics}, step=self.total_training_steps, commit=True
                )
            self.total_training_steps += 1

        return learned_activations_fired_count

    def resample_neurons(
        self,
        activation_store: TensorActivationStore,
        neuron_activity_sample_size: int,
        neuron_activity: NeuronActivity,
        train_batch_size: int,
    ) -> None:
        """Resample dead neurons.

        Args:
            activation_store: Activation store.
            neuron_activity_sample_size: Sample size for resampling.
            neuron_activity: Number of times each neuron fired.
            train_batch_size: Train batch size (also used for resampling).
        """
        if self.activation_resampler is not None:
            # Get the updates
            parameter_updates = self.activation_resampler.resample_dead_neurons(
                activation_store=activation_store,
                autoencoder=self.autoencoder,
                loss_fn=self.loss,
                neuron_activity=neuron_activity,
                train_batch_size=train_batch_size,
                neuron_activity_sample_size=neuron_activity_sample_size,
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

            # Log any metrics
            with torch.no_grad():
                metrics = {}
                if wandb.run is not None:
                    for metric in self.metrics.resample_metrics:
                        calculated = metric.calculate(
                            ResampleMetricData(
                                neuron_activity=neuron_activity,
                            )
                        )
                        metrics.update(calculated)
                    wandb.log(metrics, commit=False)

            # Reset the optimizer (TODO: Consider resetting just the relevant parameters)
            self.optimizer.reset_state_all_parameters()

    def validate_sae(self, validation_number_activations: int) -> None:
        """Get validation metrics.

        Args:
            validation_number_activations: Number of activations to use for validation.
        """
        losses: list[float] = []
        losses_with_reconstruction: list[float] = []
        losses_with_zero_ablation: list[float] = []
        source_model_device: torch.device = get_model_device(self.source_model)

        for batch in self.source_data:
            input_ids: BatchTokenizedPrompts = batch["input_ids"].to(source_model_device)

            # Run a forward pass with and without the replaced activations
            self.source_model.remove_all_hook_fns()
            replacement_hook = partial(
                replace_activations_hook, sparse_autoencoder=self.autoencoder
            )

            loss = self.source_model.forward(input_ids, return_type="loss")
            loss_with_reconstruction = self.source_model.run_with_hooks(
                input_ids,
                return_type="loss",
                fwd_hooks=[
                    (
                        self.cache_name,
                        replacement_hook,
                    )
                ],
            )
            loss_with_zero_ablation = self.source_model.run_with_hooks(
                input_ids,
                return_type="loss",
                fwd_hooks=[(self.cache_name, zero_ablate_hook)],
            )

            losses.append(loss.sum().item())
            losses_with_reconstruction.append(loss_with_reconstruction.sum().item())
            losses_with_zero_ablation.append(loss_with_zero_ablation.sum().item())

            if len(losses) >= validation_number_activations:
                break

        # Log
        validation_data = ValidationMetricData(
            source_model_loss=torch.tensor(losses),
            source_model_loss_with_reconstruction=torch.tensor(losses_with_reconstruction),
            source_model_loss_with_zero_ablation=torch.tensor(losses_with_zero_ablation),
        )
        for metric in self.metrics.validation_metrics:
            calculated = metric.calculate(validation_data)
            wandb.log(data=calculated, step=self.total_training_steps, commit=False)

    @final
    def save_checkpoint(self) -> None:
        """Save the model as a checkpoint."""
        if self.checkpoint_directory:
            file_path: Path = (
                self.checkpoint_directory / f"sae_state_dict-{self.total_training_steps}.pt"
            )
            torch.save(self.autoencoder.state_dict(), file_path)

    def run_pipeline(
        self,
        train_batch_size: int,
        max_store_size: int,
        max_activations: int,
        resample_frequency: int,
        validation_number_activations: int = 1024,
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
            validation_number_activations: Number of activations to use for validation.
            validate_frequency: Frequency at which to get validation metrics.
            checkpoint_frequency: Frequency at which to save a checkpoint.
        """
        last_resampled: int = 0
        neuron_activity_sample_size: int = 0
        last_validated: int = 0
        last_checkpoint: int = 0
        total_activations: int = 0
        neuron_activity: NeuronActivity = torch.zeros(self.autoencoder.n_learned_features)

        self.source_model.eval()  # Set the source model to evaluation (inference) mode

        # Get the store size
        store_size: int = max_store_size - max_store_size % (
            self.source_data_batch_size * self.source_dataset.context_size
        )

        with tqdm(
            desc="Activations trained on",
            total=max_activations,
        ) as progress_bar:
            for _ in range(0, max_activations, store_size):
                # Generate
                progress_bar.set_postfix({"stage": "generate"})
                activation_store: TensorActivationStore = self.generate_activations(store_size)

                # Update the counters
                num_activation_vectors_in_store = len(activation_store)
                last_resampled += num_activation_vectors_in_store
                last_validated += num_activation_vectors_in_store
                last_checkpoint += num_activation_vectors_in_store
                total_activations += num_activation_vectors_in_store
                if wandb.run is not None:
                    wandb.log({"activations_generated": total_activations}, commit=False)

                # Train
                progress_bar.set_postfix({"stage": "train"})
                batch_neuron_activity: NeuronActivity = self.train_autoencoder(
                    activation_store, train_batch_size=train_batch_size
                )
                detached_neuron_activity = batch_neuron_activity.detach().cpu()
                is_second_half_resample: bool = last_resampled > resample_frequency / 2
                if is_second_half_resample:
                    neuron_activity_sample_size += num_activation_vectors_in_store
                    neuron_activity.add_(detached_neuron_activity)

                # Resample dead neurons (if needed)
                progress_bar.set_postfix({"stage": "resample"})
                if last_resampled >= resample_frequency and self.activation_resampler is not None:
                    self.resample_neurons(
                        activation_store=activation_store,
                        neuron_activity_sample_size=neuron_activity_sample_size,
                        neuron_activity=neuron_activity,
                        train_batch_size=train_batch_size,
                    )

                    # Reset
                    last_resampled = 0
                    neuron_activity_sample_size = 0
                    neuron_activity.zero_()

                # Get validation metrics (if needed)
                progress_bar.set_postfix({"stage": "validate"})
                if validate_frequency is not None and last_validated >= validate_frequency:
                    self.validate_sae(validation_number_activations)
                    last_validated = 0

                # Checkpoint (if needed)
                progress_bar.set_postfix({"stage": "checkpoint"})
                if checkpoint_frequency is not None and last_checkpoint >= checkpoint_frequency:
                    last_checkpoint = 0
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

            >>> iterator = Pipeline.stateful_dataloader_iterable(data)
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
