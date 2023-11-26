"""Default pipeline."""
from functools import partial
from typing import final

import torch
from torch.utils.data import DataLoader
import wandb

from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
from sparse_autoencoder.metrics.validate.abstract_validate_metric import ValidationMetricData
from sparse_autoencoder.source_model.replace_activations_hook import replace_activations_hook
from sparse_autoencoder.source_model.store_activations_hook import store_activations_hook
from sparse_autoencoder.source_model.zero_ablate_hook import zero_ablate_hook
from sparse_autoencoder.tensor_types import BatchTokenizedPrompts, NeuronActivity
from sparse_autoencoder.train.abstract_pipeline import AbstractPipeline
from sparse_autoencoder.train.utils import get_model_device


@final
class Pipeline(AbstractPipeline):
    """Pipeline for training a Sparse Autoencoder on TransformerLens activations.

    Includes all the key functionality to train a sparse autoencoder, with a specific set of
        hyperparameters.
    """

    total_training_steps: int = 1

    def generate_activations(self, store_size: int) -> TensorActivationStore:
        """Generate activations.

        Args:
            store_size: Number of activations to generate.

        Returns:
            Activation store for the train section.
        """
        num_neurons: int = self.autoencoder.n_input_features
        source_model_device: torch.device = get_model_device(self.source_model)

        store = TensorActivationStore(store_size, num_neurons)

        # Set model to evaluation (inference) mode
        self.source_model.eval()

        # Add the hook to the model (will automatically store the activations every time the model
        # runs)
        self.source_model.remove_all_hook_fns()
        hook = partial(store_activations_hook, store=store)
        self.source_model.add_hook(self.cache_name, hook)

        # Loop through the dataloader until the store reaches the desired size
        with torch.no_grad():
            for batch in self.source_data:
                input_ids: BatchTokenizedPrompts = batch["input_ids"].to(source_model_device)
                self.source_model.forward(input_ids, stop_at_layer=self.layer + 1)  # type: ignore (TLens is typed incorrectly)

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
