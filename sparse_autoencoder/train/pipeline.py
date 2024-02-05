"""Default pipeline."""
from collections.abc import Iterator
from functools import partial
import logging
from pathlib import Path
from tempfile import gettempdir
from typing import TYPE_CHECKING, final

from jaxtyping import Float, Int
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from pydantic import NonNegativeInt, PositiveInt, validate_call
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import wandb

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.lightning import LitSparseAutoencoder
from sparse_autoencoder.metrics.validate.reconstruction_score import ReconstructionScoreMetric
from sparse_autoencoder.metrics.wrappers.classwise import ClasswiseWrapperWithMean
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TorchTokenizedPrompts
from sparse_autoencoder.source_model.replace_activations_hook import replace_activations_hook
from sparse_autoencoder.source_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.source_model.zero_ablate_hook import zero_ablate_hook
from sparse_autoencoder.train.utils.get_model_device import get_model_device
from sparse_autoencoder.utils.data_parallel import DataParallelWithModelAttributes


if TYPE_CHECKING:
    from sparse_autoencoder.tensor_types import Axis


DEFAULT_CHECKPOINT_DIRECTORY: Path = Path(gettempdir()) / "sparse_autoencoder"


class Pipeline:
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    autoencoder: LitSparseAutoencoder
    """Sparse autoencoder to train."""

    n_input_features: int
    """Number of input features in the sparse autoencoder."""

    n_learned_features: int
    """Number of learned features in the sparse autoencoder."""

    cache_names: list[str]
    """Names of the cache hook points to use in the source model."""

    layer: int
    """Layer to stope the source model at (if we don't need activations after this layer)."""

    log_frequency: int
    """Frequency at which to log metrics (in steps)."""

    progress_bar: tqdm | None
    """Progress bar for the pipeline."""

    source_data: Iterator[TorchTokenizedPrompts]
    """Iterable over the source data."""

    source_dataset: SourceDataset
    """Source dataset to generate activation data from (tokenized prompts)."""

    source_model: HookedTransformer | DataParallelWithModelAttributes[HookedTransformer]
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
        autoencoder: LitSparseAutoencoder,
        cache_names: list[str],
        layer: NonNegativeInt,
        source_dataset: SourceDataset,
        source_model: HookedTransformer | DataParallelWithModelAttributes[HookedTransformer],
        n_input_features: int,
        n_learned_features: int,
        run_name: str = "sparse_autoencoder",
        checkpoint_directory: Path = DEFAULT_CHECKPOINT_DIRECTORY,
        log_frequency: PositiveInt = 100,
        num_workers_data_loading: NonNegativeInt = 0,
        source_data_batch_size: PositiveInt = 12,
    ) -> None:
        """Initialize the pipeline.

        Args:
            autoencoder: Sparse autoencoder to train.
            cache_names: Names of the cache hook points to use in the source model.
            layer: Layer to stope the source model at (if we don't need activations after this
                layer).
            source_dataset: Source dataset to get data from.
            source_model: Source model to get activations from.
            n_input_features: Number of input features in the sparse autoencoder.
            n_learned_features: Number of learned features in the sparse autoencoder.
            run_name: Name of the run for saving checkpoints.
            checkpoint_directory: Directory to save checkpoints to.
            log_frequency: Frequency at which to log metrics (in steps)
            num_workers_data_loading: Number of CPU workers for the dataloader.
            source_data_batch_size: Batch size for the source data.
        """
        self.autoencoder = autoencoder
        self.cache_names = cache_names
        self.checkpoint_directory = checkpoint_directory
        self.layer = layer
        self.log_frequency = log_frequency
        self.run_name = run_name
        self.source_data_batch_size = source_data_batch_size
        self.source_dataset = source_dataset
        self.source_model = source_model
        self.n_input_features = n_input_features
        self.n_learned_features = n_learned_features

        # Add validate metric
        self.reconstruction_score = ClasswiseWrapperWithMean(
            ReconstructionScoreMetric(len(cache_names)),
            component_names=cache_names,
            prefix="validation/reconstruction_score",
        )
        self.reconstruction_score.to(get_model_device(self.autoencoder))

        # Create a stateful iterator
        source_dataloader = source_dataset.get_dataloader(
            source_data_batch_size, num_workers=num_workers_data_loading
        )
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
        source_model_device = get_model_device(self.source_model)
        store = TensorActivationStore(
            store_size, self.n_input_features, n_components=self.n_components
        )

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

    def train_autoencoder(
        self,
        activation_store: TensorActivationStore,
        train_batch_size: PositiveInt,
    ) -> None:
        """Train the sparse autoencoder.

        Args:
            activation_store: Activation store from the generate section.
            train_batch_size: Train batch size.

        Returns:
            Number of times each neuron fired, for each component.
        """
        activations_dataloader = DataLoader(
            activation_store, batch_size=train_batch_size, num_workers=4, persistent_workers=False
        )

        # Setup the trainer with no console logging
        logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
        trainer = Trainer(
            logger=WandbLogger() if wandb.run is not None else None,
            max_epochs=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            precision="16-mixed",
        )
        trainer.fit(self.autoencoder, activations_dataloader)

    @validate_call
    def validate_sae(self, validation_n_activations: PositiveInt) -> None:
        """Get validation metrics.

        Args:
            validation_n_activations: Number of activations to use for validation.
        """
        n_batches = validation_n_activations // (
            self.source_data_batch_size * self.source_dataset.context_size
        )
        source_model_device = get_model_device(self.source_model)

        # Create the metric data stores
        losses: Float[Tensor, Axis.COMPONENT] = torch.zeros(
            self.n_components, device=source_model_device
        )
        losses_with_reconstruction: Float[Tensor, Axis.COMPONENT] = torch.zeros(
            self.n_components, device=source_model_device
        )
        losses_with_zero_ablation: Float[Tensor, Axis.COMPONENT] = torch.zeros(
            self.n_components, device=source_model_device
        )

        sae_model = self.autoencoder.sparse_autoencoder.clone()
        sae_model.to(source_model_device)

        for component_idx, cache_name in enumerate(self.cache_names):
            for _batch_idx in range(n_batches):
                batch = next(self.source_data)

                input_ids: Int[Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)] = batch[
                    "input_ids"
                ].to(source_model_device)

                # Run a forward pass with and without the replaced activations
                self.source_model.remove_all_hook_fns()
                replacement_hook = partial(
                    replace_activations_hook,
                    sparse_autoencoder=sae_model,
                    component_idx=component_idx,
                    n_components=self.n_components,
                )

                with torch.no_grad():
                    loss: Float[
                        Tensor, Axis.names(Axis.SOURCE_DATA_BATCH, Axis.POSITION)
                    ] = self.source_model.forward(input_ids, return_type="loss")
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
                        input_ids, return_type="loss", fwd_hooks=[(cache_name, zero_ablate_hook)]
                    )

                    self.reconstruction_score.update(
                        source_model_loss=loss,
                        source_model_loss_with_reconstruction=loss_with_reconstruction,
                        source_model_loss_with_zero_ablation=loss_with_zero_ablation,
                        component_idx=component_idx,
                    )

                    losses[component_idx] += loss.sum()
                    losses_with_reconstruction[component_idx] += loss_with_reconstruction.sum()
                    losses_with_zero_ablation[component_idx] += loss_with_zero_ablation.sum()

        # Log
        if wandb.run is not None:
            log = {
                f"validation/source_model_losses/{c}": val
                for c, val in zip(self.cache_names, losses / n_batches)
            }
            log.update(
                {
                    f"validation/source_model_losses_with_reconstruction/{c}": val
                    for c, val in zip(self.cache_names, losses_with_reconstruction / n_batches)
                }
            )
            log.update(
                {
                    f"validation/source_model_losses_with_zero_ablation/{c}": val
                    for c, val in zip(self.cache_names, losses_with_zero_ablation / n_batches)
                }
            )
            log.update(self.reconstruction_score.compute())
            wandb.log(log)

    @final
    def save_checkpoint(self, *, is_final: bool = False) -> Path:
        """Save the model as a checkpoint.

        Args:
            is_final: Whether this is the final checkpoint.

        Returns:
            Path to the saved checkpoint.
        """
        name: str = f"{self.run_name}_{'final' if is_final else self.total_activations_trained_on}"

        # Wandb
        if wandb.run is not None:
            self.autoencoder.sparse_autoencoder.save_to_wandb(name)

        # Local
        local_path = self.checkpoint_directory / f"{name}.pt"
        self.autoencoder.sparse_autoencoder.save(local_path)
        return local_path

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

        # Get the loss fn
        loss_fn = self.autoencoder.loss_fn.clone()
        loss_fn.keep_batch_dim = True

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

                # Train & resample if needed
                progress_bar.set_postfix({"stage": "train"})
                self.train_autoencoder(activation_store, train_batch_size=train_batch_size)

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
