"""Training Pipeline."""
from torch import device, set_grad_enabled
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from sparse_autoencoder.activation_store.base_store import ActivationStoreItem
from sparse_autoencoder.autoencoder.loss import (
    l1_loss,
    reconstruction_loss,
    sae_training_loss,
)
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime


def train_autoencoder(
    activations_dataloader: DataLoader[ActivationStoreItem],
    autoencoder: SparseAutoencoder,
    optimizer: Optimizer,
    sweep_parameters: SweepParametersRuntime,
    log_interval: int = 10,
    device: device | None = None,
) -> None:
    """Sparse Autoencoder Training Loop.

    Args:
        activations_dataloader: DataLoader containing activations.
        autoencoder: Sparse autoencoder model.
        optimizer: The optimizer to use.
        sweep_parameters: The sweep parameters to use.
        log_interval: How often to log progress.
        device: Decide to use.
    """
    n_dataset_items: int = len(activations_dataloader.dataset)  # type: ignore
    batch_size: int = activations_dataloader.batch_size  # type: ignore

    with set_grad_enabled(True), tqdm(  # noqa: FBT003
        desc="Train Autoencoder",
        total=n_dataset_items,
        colour="green",
        position=1,
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
            reconstruction_loss_mse = reconstruction_loss(
                batch,
                reconstructed_activations,
            )
            l1_loss_learned_activations = l1_loss(learned_activations)
            total_loss = sae_training_loss(
                reconstruction_loss_mse,
                l1_loss_learned_activations,
                sweep_parameters.l1_coefficient,
            )

            # TODO: Store the learned activations (default every 25k steps)

            # Backwards pass
            total_loss.backward()

            optimizer.step()

            # Log
            if step % log_interval == 0 and wandb.run is not None:
                wandb.log(
                    {
                        "reconstruction_loss": reconstruction_loss_mse,
                        "l1_loss": l1_loss_learned_activations,
                        "loss": total_loss,
                    },
                )

            # TODO: Get the feature density & also log to wandb

            # TODO: Apply neuron resampling if enabled

            progress_bar.update(batch_size)

        progress_bar.close()
