"""Training Pipeline."""
from dataclasses import dataclass

import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

import wandb
from sparse_autoencoder.activations.ListActivationStore import ListActivationStore
from sparse_autoencoder.autoencoder.loss import (
    l1_loss,
    reconstruction_loss,
    sae_training_loss,
)
from sparse_autoencoder.autoencoder.model import SparseAutoencoder


def resample_neurons():
    """Resample Neurons.

    TODO: Implement this function."""
    pass


@dataclass
class TrainChunkSteps:
    """Training Statistics."""

    reconstruction_loss: float
    l1_loss: float
    total_loss: float
    steps: int


def train(
    autoencoder: SparseAutoencoder,
    activation_buffer: ListActivationStore,
    l1_coefficient: float,
    min_buffer: int = 100,
) -> TrainChunkSteps:
    """Train the Sparse AutoEncoder"""
    optimizer = Adam(autoencoder.parameters())

    steps = len(activation_buffer) - min_buffer

    for _ in range(steps):
        print("step", _)
        sample_batch = activation_buffer.sample_without_replace(4)

        # Forward pass
        learned_activations, reconstructed_activations = autoencoder(sample_batch)
        reconstruction_loss_mse = reconstruction_loss(
            sample_batch, reconstructed_activations
        )
        l1_loss_learned_activations = l1_loss(learned_activations)
        total_loss = sae_training_loss(
            reconstruction_loss_mse, l1_loss_learned_activations, l1_coefficient
        )

        # Backwards pass
        total_loss.backward()
        optimizer.step()

        # Zero the gradients
        optimizer.zero_grad()

    return TrainChunkSteps(
        reconstruction_loss=reconstruction_loss_mse.sum().item(),
        l1_loss=l1_loss_learned_activations.sum().item(),
        total_loss=total_loss.item(),
        steps=steps,
    )


def pipeline(
    src_model: HookedTransformer,
    autoencoder: SparseAutoencoder,
    activation_hook_point: str,
    prompts_dataloader: DataLoader,
    min_buffer: int = 100,
    max_buffer: int = 200,
    l1_coefficient: float = 0.006,
):
    """Full pipeline for training the Sparse AutoEncoder"""
    device = transformer_lens.utils.get_device()
    src_model.to(device)
    autoencoder.to(device)

    # Create the activation buffer
    activation_buffer = ListActivationStore()
    buffer_size: int = len(activation_buffer)

    # TODO: Hook the transformer to get just the cache item we want. Also kill any later layers as
    # we don't need the logits

    # Setup wandb
    wandb.init(
        config={
            "input_features": autoencoder.n_input_features,
            "learned_features": autoencoder.n_learned_features,
            "activation_hook_point": activation_hook_point,
            "min_buffer": min_buffer,
            "max_buffer": max_buffer,
            "l1_coefficient": l1_coefficient,
        }
    )
    steps: int = 0

    # Whilst there is still data
    while True:
        # Whilst the buffer size is not full
        if buffer_size <= max_buffer:
            needed = max_buffer - buffer_size

            # Sample a prompt from the dataset
            for idx, (input_ids, attention_mask) in enumerate(prompts_dataloader):
                # Run a forward pass and get the activations
                with torch.no_grad():
                    _logits, cache = src_model.run_with_cache(input_ids.to(device))
                    activations: Float[Tensor, "batch pos activations"] = cache[
                        activation_hook_point  # TODO: Flatten if e.g. 4 axis (from head_idx)
                    ]

                    # For each batch item, get the non padded tokens
                    tokens_added: int = 0
                    for batch_idx, batch_item in enumerate(activations):
                        non_padded_tokens: Float[
                            Tensor, "pos activations"
                        ] = batch_item[attention_mask[batch_idx] == 1]

                        tokens_added += len(non_padded_tokens)

                        # Store the activations in the buffer
                        activation_buffer.append(non_padded_tokens)

                    wandb.log({"tokens_added": tokens_added}, steps)

                # Move on once we have enough
                if idx >= needed:
                    break

            # Stop if we no longer add any more
            if len(activation_buffer) == 0:
                break

        # Whilst the buffer is less than the minimum, train the autoencoder
        print("training autoencoder")
        stats = train(autoencoder, activation_buffer, l1_coefficient, min_buffer)
        steps += stats.steps

        wandb.log(
            {
                "reconstruction_loss": stats.reconstruction_loss,
                "l1_loss": stats.l1_loss,
                "total_loss": stats.total_loss,
            },
            steps,
        )
