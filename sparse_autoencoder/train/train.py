"""Training Pipeline."""
import torch
import transformer_lens
from jaxtyping import Float
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from sparse_autoencoder.activations.ActivationBuffer import ActivationBuffer
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


def train(
    autoencoder: SparseAutoencoder,
    activation_buffer: ActivationBuffer,
    l1_coefficient: float,
    min_buffer: int = 100,
):
    """Train the Sparse AutoEncoder"""
    optimizer = Adam(autoencoder.parameters())

    for _ in range(len(activation_buffer) - min_buffer):
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
    activation_buffer = ActivationBuffer()
    buffer_size: int = len(activation_buffer)

    # TODO: Hook the transformer to get just the cache item we want

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
                    for batch_idx, batch_item in enumerate(activations):
                        non_padded_tokens: Float[
                            Tensor, "pos activations"
                        ] = batch_item[attention_mask[batch_idx] == 1]

                        # Store the activations in the buffer
                        activation_buffer.append(non_padded_tokens)

                # Move on once we have enough
                if idx >= needed:
                    break

            # Stop if we no longer add any more
            if len(activation_buffer) == 0:
                break

        # Whilst the buffer is less than the minimum, train the autoencoder
        train(autoencoder, activation_buffer, l1_coefficient, min_buffer)
