"""Training Pipeline."""
import torch
from torch import device, set_grad_enabled
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.autoencoder.loss import (
    l1_loss,
    reconstruction_loss,
    sae_training_loss,
)
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.tensor_types import BatchLoss, LearnedFeatures, NeuronActivity
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime


def train_autoencoder(
    activation_store: ActivationStore,
    autoencoder: SparseAutoencoder,
    optimizer: Optimizer,
    sweep_parameters: SweepParametersRuntime,
    previous_steps: int,
    log_interval: int = 10,
    device: device | None = None,
) -> tuple[int, LearnedFeatures]:
    """Sparse Autoencoder Training Loop.

    Args:
        activation_store: Activation store to train on.
        autoencoder: Sparse autoencoder model.
        optimizer: The optimizer to use.
        sweep_parameters: The sweep parameters to use.
        previous_steps: Training steps from previous generate/train iterations.
        log_interval: How often to log progress.
        device: Decide to use.

    Returns:
        Number of steps taken.
    """
    # Create a dataloader from the store
    activations_dataloader = DataLoader(
        activation_store,
        batch_size=sweep_parameters.batch_size,
    )

    n_dataset_items: int = len(activation_store)
    batch_size: int = sweep_parameters.batch_size

    learned_activations_fired_count: NeuronActivity = torch.zeros(
        autoencoder.n_learned_features, dtype=torch.int32, device=device
    )

    step = 0
    with set_grad_enabled(True), tqdm(  # noqa: FBT003
        desc="Train Autoencoder",
        total=n_dataset_items,
        colour="green",
        leave=False,
        dynamic_ncols=True,
    ) as progress_bar:
        for step, batch in enumerate(activations_dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Move the batch to the device (in place)
            batch = batch.to(device)  # noqa: PLW2901

            # Forward pass
            learned_activations, reconstructed_activations = autoencoder(batch)

            # Get metrics
            reconstruction_loss_mse: BatchLoss = reconstruction_loss(
                batch,
                reconstructed_activations,
            )
            l1_loss_learned_activations: BatchLoss = l1_loss(learned_activations)
            total_loss: BatchLoss = sae_training_loss(
                reconstruction_loss_mse,
                l1_loss_learned_activations,
                sweep_parameters.l1_coefficient,
            )

            # Store count of how many neurons have fired
            fired = learned_activations > 0
            learned_activations_fired_count.add_(fired.sum(dim=0))

            # Backwards pass
            total_loss.mean().backward()

            optimizer.step()

            # Log
            if step % log_interval == 0 and wandb.run is not None:
                wandb.log(
                    {
                        "reconstruction_loss": reconstruction_loss_mse.mean().item(),
                        "l1_loss": l1_loss_learned_activations.mean().item(),
                        "loss": total_loss.mean().item(),
                    },
                )

            progress_bar.update(batch_size)

        current_step = previous_steps + step + 1
        return current_step, learned_activations_fired_count
