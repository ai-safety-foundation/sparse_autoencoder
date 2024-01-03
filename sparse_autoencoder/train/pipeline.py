"""Default pipeline."""
from collections.abc import Iterator
from functools import partial
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, final
from urllib.parse import quote_plus

from jaxtyping import Float, Int, Int64
from pydantic import NonNegativeInt, PositiveInt, validate_call
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import wandb

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
    ParameterUpdateResults,
)
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossReductionType
from sparse_autoencoder.metrics.metrics_container import MetricsContainer, default_metrics
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.validate.abstract_validate_metric import ValidationMetricData
from sparse_autoencoder.optimizer.abstract_optimizer import AbstractOptimizerWithReset
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TorchTokenizedPrompts
from sparse_autoencoder.source_model.replace_activations_hook import replace_activations_hook
from sparse_autoencoder.source_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.source_model.zero_ablate_hook import zero_ablate_hook
from sparse_autoencoder.tensor_types import Axis
from sparse_autoencoder.train.utils import get_model_device


if TYPE_CHECKING:
    from sparse_autoencoder.metrics.abstract_metric import MetricResult

DEFAULT_CHECKPOINT_DIRECTORY: Path = Path(tempfile.gettempdir()) / "sparse_autoencoder"


class Pipeline:
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    activation_resampler: AbstractActivationResampler | None
    """Activation resampler to use."""

    autoencoder: SparseAutoencoder
    """Sparse autoencoder to train."""

    cache_names: list[str]
    """Names of the cache hook points to use in the source model."""

    layer: int
    """Layer to stope the source model at (if we don't need activations after this layer)."""

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

    source_data: Iterator[TorchTokenizedPrompts]
    """Iterable over the source data."""

    source_dataset: SourceDataset
    """Source dataset to generate activation data from (tokenized prompts)."""

    source_model: HookedTransformer
    """Source model to get activations from."""

    total_activations_trained_on: int = 0
    """Total number of activations trained on state."""

    @property
    def n_components(self) -> int:
        """Number of source model components the SAE is trained on."""
        return len(self.cache_names)

    @final
    @validate_call(config={"arbitrary_types_allowed": True})
    def __init__(
        self,
        activation_resampler: AbstractActivationResampler | None,
        autoencoder: SparseAutoencoder,
        cache_names: list[str],
        layer: NonNegativeInt,
        loss: AbstractLoss,
        optimizer: AbstractOptimizerWithReset,
        source_dataset: SourceDataset,
        source_model: HookedTransformer,
        run_name: str = "sparse_autoencoder",
        checkpoint_directory: Path = DEFAULT_CHECKPOINT_DIRECTORY,
        log_frequency: PositiveInt = 100,
        metrics: MetricsContainer = default_metrics,
        source_data_batch_size: PositiveInt = 12,
    ) -> None:
        """Initialize the pipeline.

        Args:
            activation_resampler: Activation resampler to use.
            autoencoder: Sparse autoencoder to train.
            cache_names: Names of the cache hook points to use in the source model.
            layer: Layer to stope the source model at (if we don't need activations after this
                layer).
            loss: Loss function to use.
            optimizer: Optimizer to use.
            source_dataset: Source dataset to get data from.
            source_model: Source model to get activations from.
            run_name: Name of the run for saving checkpoints.
            checkpoint_directory: Directory to save checkpoints to.
            log_frequency: Frequency at which to log metrics (in steps)
            metrics: Metrics to use.
            source_data_batch_size: Batch size for the source data.
        """
        self.activation_resampler = activation_resampler
        self.autoencoder = autoencoder
        self.cache_names = cache_names
        self.checkpoint_directory = checkpoint_directory
        self.layer = layer
        self.log_frequency = log_frequency
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.run_name = run_name
        self.source_data_batch_size = source_data_batch_size
        self.source_dataset = source_dataset
        self.source_model = source_model

        # Create a stateful iterator
        source_dataloader = source_dataset.get_dataloader(source_data_batch_size)
        self.source_data = iter(source_dataloader)

    @validate_call
    def generate_activations(self, store_size: PositiveInt) -> TensorActivationStore:
        """Generate activations.

        Args:
            store_size: Number of activations to generate.

        Returns:
            Activation store for the train section.

        Raises:
            ValueError: If the store size is not divisible by the batch size.
        """
        # Check the store size is divisible by the batch size
        if store_size % (self.source_data_batch_size * self.source_dataset.context_size) != 0:
            error_message = (
                f"Store size must be divisible by the batch size ({self.source_data_batch_size}), "
                f"got {store_size}"
            )
            raise ValueError(error_message)

        # Setup the store
        n_neurons: int = self.autoencoder.n_input_features
        source_model_device: torch.device = get_model_device(self.source_model)
        store = TensorActivationStore(store_size, n_neurons, n_components=self.n_components)

        # Add the hook to the model (will automatically store the activations every time the model
        # runs)
        self.source_model.remove_all_hook_fns()
        for component_idx, cache_name in enumerate(self.cache_names):
            hook = partial(store_activations_hook, store=store, component_idx=component_idx)
            self.source_model.add_hook(cache_name, hook)

        # Loop through the dataloader until the store reaches the desired size
        with torch.no_grad():
            while len(store) < store_size:
                batch = next(self.source_data)
                input_ids: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)] = batch[
                    "input_ids"
                ].to(source_model_device)
                self.source_model.forward(
                    input_ids, stop_at_layer=self.layer + 1, prepend_bos=False
                )  # type: ignore (TLens is typed incorrectly)

        self.source_model.remove_all_hook_fns()
        store.shuffle()

        return store

    @validate_call(config={"arbitrary_types_allowed": True})
    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: PositiveInt
    ) -> Int64[Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)]:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Number of times each neuron fired, for each component.
        """
        autoencoder_device: torch.device = get_model_device(self.autoencoder)

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
        )

        learned_activations_fired_count: Int64[
            Tensor, Axis.names(Axis.COMPONENT, Axis.LEARNT_FEATURE)
        ] = torch.zeros(
            (self.n_components, self.autoencoder.n_learned_features),
            dtype=torch.int64,
            device=autoencoder_device,
        )

        for store_batch in activations_dataloader:
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(autoencoder_device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder.forward(batch)

            # Get loss & metrics
            metrics: list[MetricResult] = []
            total_loss, loss_metrics = self.loss.scalar_loss_with_log(
                batch,
                learned_activations,
                reconstructed_activations,
                component_reduction=LossReductionType.MEAN,
            )
            metrics.extend(loss_metrics)

            with torch.no_grad():
                for metric in self.metrics.train_metrics:
                    calculated = metric.calculate(
                        TrainMetricData(batch, learned_activations, reconstructed_activations)
                    )
                    metrics.extend(calculated)

            # Store count of how many neurons have fired
            with torch.no_grad():
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))

            # Backwards pass
            total_loss.backward()
            self.optimizer.step()
            self.autoencoder.post_backwards_hook()

            # Log training metrics
            self.total_activations_trained_on += train_batch_size
            if (
                wandb.run is not None
                and int(self.total_activations_trained_on / train_batch_size) % self.log_frequency
                == 0
            ):
                log = {}
                for metric_result in metrics:
                    log.update(metric_result.wandb_log)
                wandb.log(
                    log,
                    step=self.total_activations_trained_on,
                    commit=True,
                )

        return learned_activations_fired_count

    def update_parameters(self, parameter_updates: list[ParameterUpdateResults]) -> None:
        """Update the parameters of the model from the results of the resampler.

        Args:
            parameter_updates: Parameter updates from the resampler.
        """
        for component_idx, component_parameter_update in enumerate(parameter_updates):
            # Update the weights and biases
            self.autoencoder.encoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_weight_updates,
                component_idx=component_idx,
            )
            self.autoencoder.encoder.update_bias(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_encoder_bias_updates,
                component_idx=component_idx,
            )
            self.autoencoder.decoder.update_dictionary_vectors(
                component_parameter_update.dead_neuron_indices,
                component_parameter_update.dead_decoder_weight_updates,
                component_idx=component_idx,
            )

            # Reset the optimizer
            for parameter, axis in self.autoencoder.reset_optimizer_parameter_details:
                self.optimizer.reset_neurons_state(
                    parameter=parameter,
                    neuron_indices=component_parameter_update.dead_neuron_indices,
                    axis=axis,
                    component_idx=component_idx,
                )

    @validate_call
    def validate_sae(self, validation_n_activations: PositiveInt) -> None:
        """Get validation metrics.

        Args:
            validation_n_activations: Number of activations to use for validation.
        """
        losses_shape = (
            validation_n_activations // self.source_data_batch_size,
            self.n_components,
        )
        source_model_device: torch.device = get_model_device(self.source_model)

        # Create the metric data stores
        losses: Float[Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT)] = torch.empty(
            losses_shape, device=source_model_device
        )
        losses_with_reconstruction: Float[
            Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT)
        ] = torch.empty(losses_shape, device=source_model_device)
        losses_with_zero_ablation: Float[
            Tensor, Axis.names(Axis.ITEMS, Axis.COMPONENT)
        ] = torch.empty(losses_shape, device=source_model_device)

        for component_idx, cache_name in enumerate(self.cache_names):
            for batch_idx in range(losses.shape[0]):
                batch = next(self.source_data)

                input_ids: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)] = batch[
                    "input_ids"
                ].to(source_model_device)

                # Run a forward pass with and without the replaced activations
                self.source_model.remove_all_hook_fns()
                replacement_hook = partial(
                    replace_activations_hook,
                    sparse_autoencoder=self.autoencoder,
                    component_idx=component_idx,
                )

                with torch.no_grad():
                    loss = self.source_model.forward(input_ids, return_type="loss")
                    loss_with_reconstruction = self.source_model.run_with_hooks(
                        input_ids,
                        return_type="loss",
                        fwd_hooks=[
                            (
                                cache_name,
                                replacement_hook,
                            )
                        ],
                    )
                    loss_with_zero_ablation = self.source_model.run_with_hooks(
                        input_ids,
                        return_type="loss",
                        fwd_hooks=[(cache_name, zero_ablate_hook)],
                    )

                    losses[batch_idx, component_idx] = loss.sum()
                    losses_with_reconstruction[
                        batch_idx, component_idx
                    ] = loss_with_reconstruction.sum()
                    losses_with_zero_ablation[
                        batch_idx, component_idx
                    ] = loss_with_zero_ablation.sum()

        # Log
        validation_data = ValidationMetricData(
            source_model_loss=losses,
            source_model_loss_with_reconstruction=losses_with_reconstruction,
            source_model_loss_with_zero_ablation=losses_with_zero_ablation,
        )
        for metric in self.metrics.validation_metrics:
            log = {}
            for metric_result in metric.calculate(validation_data):
                log.update(metric_result.wandb_log)
            if wandb.run is not None:
                wandb.log(log, commit=False)

    @final
    def save_checkpoint(self, *, is_final: bool = False) -> Path:
        """Save the model as a checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        # Create the name
        name: str = f"{self.run_name}_{'final' if is_final else self.total_activations_trained_on}"
        safe_name = quote_plus(name, safe="_")

        # Save locally
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        file_path: Path = self.checkpoint_directory / f"{safe_name}.pt"
        torch.save(
            self.autoencoder.state_dict(),
            file_path,
        )

        # Upload to wandb
        if wandb.run is not None:
            artifact = wandb.Artifact(safe_name, type="model")
            artifact.add_file(str(file_path))
            wandb.log_artifact(artifact)

        return file_path

    @validate_call
    def run_pipeline(
        self,
        train_batch_size: PositiveInt,
        max_store_size: PositiveInt,
        max_activations: PositiveInt,
        validation_n_activations: PositiveInt = 1024,
        validate_frequency: PositiveInt | None = None,
        checkpoint_frequency: PositiveInt | None = None,
    ) -> None:
        """Run the full training pipeline.

        Args:
            train_batch_size: Train batch size.
            max_store_size: Maximum size of the activation store.
            max_activations: Maximum total number of activations to train on (the original paper
                used 8bn, although others have had success with 100m+).
            validation_n_activations: Number of activations to use for validation.
            validate_frequency: Frequency at which to get validation metrics.
            checkpoint_frequency: Frequency at which to save a checkpoint.
        """
        last_validated: int = 0
        last_checkpoint: int = 0

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
                n_activation_vectors_in_store = len(activation_store)
                last_validated += n_activation_vectors_in_store
                last_checkpoint += n_activation_vectors_in_store

                # Train
                progress_bar.set_postfix({"stage": "train"})
                batch_neuron_activity: Int64[Tensor, Axis.LEARNT_FEATURE] = self.train_autoencoder(
                    activation_store, train_batch_size=train_batch_size
                )

                # Resample dead neurons (if needed)
                progress_bar.set_postfix({"stage": "resample"})
                if self.activation_resampler is not None:
                    # Get the updates
                    parameter_updates = self.activation_resampler.step_resampler(
                        batch_neuron_activity=batch_neuron_activity,
                        activation_store=activation_store,
                        autoencoder=self.autoencoder,
                        loss_fn=self.loss,
                        train_batch_size=train_batch_size,
                    )

                    if parameter_updates is not None:
                        if wandb.run is not None:
                            wandb.log(
                                {
                                    "resample/dead_neurons": [
                                        len(update.dead_neuron_indices)
                                        for update in parameter_updates
                                    ]
                                },
                                commit=False,
                            )

                        # Update the parameters
                        self.update_parameters(parameter_updates)

                # Get validation metrics (if needed)
                progress_bar.set_postfix({"stage": "validate"})
                if validate_frequency is not None and last_validated >= validate_frequency:
                    self.validate_sae(validation_n_activations)
                    last_validated = 0

                # Checkpoint (if needed)
                progress_bar.set_postfix({"stage": "checkpoint"})
                if checkpoint_frequency is not None and last_checkpoint >= checkpoint_frequency:
                    last_checkpoint = 0
                    self.save_checkpoint()

                # Update the progress bar
                progress_bar.update(store_size)

        # Save the final checkpoint
        self.save_checkpoint(is_final=True)
