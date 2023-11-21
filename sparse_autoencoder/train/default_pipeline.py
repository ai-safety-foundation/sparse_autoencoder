"""Default pipeline."""
from functools import partial
from typing import final

import torch
from torch.utils.data import DataLoader
import wandb

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    ParameterUpdateResults,
)
from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.src_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.tensor_types import BatchTokenizedPrompts, NeuronActivity
from sparse_autoencoder.train.abstract_pipeline import AbstractPipeline
from sparse_autoencoder.train.resample_neurons import resample_dead_neurons
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime


@final
class DefaultPipeline(AbstractPipeline):
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    total_training_steps: int = 1

    def generate_activations(self, store_size: int) -> TensorActivationStore:
        """Generate activations."""
        num_neurons: int = 256
        model_device: torch.device = torch.device("mps")

        store = TensorActivationStore(store_size, num_neurons)

        # Set model to evaluation (inference) mode
        self.source_model.eval()
        self.source_model.to(model_device, print_details=False)

        # Add the hook to the model (will automatically store the activations every time the model runs)
        self.source_model.remove_all_hook_fns()
        hook = partial(store_activations_hook, store=store)
        self.source_model.add_hook(self.cache_name, hook)

        # Loop through the dataloader until the store reaches the desired size
        with torch.no_grad():
            for batch in self.source_data:
                input_ids: BatchTokenizedPrompts = batch["input_ids"].to(model_device)
                self.source_model.forward(input_ids, stop_at_layer=self.layer + 1)  # type: ignore (TLens is typed incorrectly)

                if len(store) >= store_size:
                    break

        self.source_model.remove_all_hook_fns()
        store.shuffle()

        return store

    def train_autoencoder(
        self, activation_store: TensorActivationStore, train_batch_size: int
    ) -> NeuronActivity:
        """Train the sparse autoencoder."""
        # TODO: move these to config
        model_device: torch.device = torch.device("mps")

        activations_dataloader = DataLoader(
            activation_store,
            batch_size=train_batch_size,
        )

        learned_activations_fired_count: NeuronActivity = torch.zeros(
            self.autoencoder.n_learned_features, dtype=torch.int32, device=model_device
        )

        for store_batch in activations_dataloader:
            # Zero the gradients
            self.optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = store_batch.detach().to(model_device)

            # Forward pass
            learned_activations, reconstructed_activations = self.autoencoder(batch)

            # Get metrics
            total_loss, loss_metrics = self.loss.batch_scalar_loss_with_log(
                batch, learned_activations, reconstructed_activations
            )

            # for metric in self.train_metrics:
            # TODO: Handle train metrics

            # Store count of how many neurons have fired
            with torch.no_grad():
                fired = learned_activations > 0
                learned_activations_fired_count.add_(fired.sum(dim=0))

            # Backwards pass
            total_loss.backward()
            self.optimizer.step()

            # Log
            # TODO: Make this commit at some point
            # if step % log_interval == 0 and wandb.run is not None:
            if wandb.run is not None:
                wandb.log(data=loss_metrics, step=self.total_training_steps)
            self.total_training_steps += 1

        return learned_activations_fired_count

    def validate_sae(self) -> None:
        """Get validation metrics."""
        # Not currently setup
        return

    def save_checkpoint(self) -> None:
        """Save the model as a checkpoint."""
        # Not currently setup
        return
