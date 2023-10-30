from functools import partial

import torch
from jaxtyping import Float
from torch import Tensor
from torch.optim import Adam
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer

from sparse_autoencoder.activations.ActivationBuffer import ActivationBuffer
from sparse_autoencoder.autoencoder.loss import (
    l1_loss,
    reconstruction_loss,
    sae_training_loss,
)
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.dataset.dataloader import collate_pile, create_dataloader


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
        sample_batch = activation_buffer.sample_without_replace(8192)

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
    min_buffer: int = 100,
    max_buffer: int = 200,
    l1_coefficient: float = 0.006,
):
    """Full pipeline for training the Sparse AutoEncoder"""

    # Create the prompts dataloader
    tokenizer = src_model.tokenizer
    collate_fn = partial(collate_pile, tokenizer=tokenizer)
    prompts_dataloader = create_dataloader(
        "monology/pile-uncopyrighted",
        collate_fn,
    )

    # Create the activation buffer
    activation_buffer = ActivationBuffer()
    buffer_size: int = len(activation_buffer)

    # TODO: Hook the transformer to get just the cache item we want

    # Whilst there are batches available
    while buffer_size > 0:
        # Whilst the buffer size is not full
        if buffer_size <= max_buffer:
            needed = max_buffer - buffer_size

            # Sample a prompt from the dataset
            for idx, (input_ids, attention_mask) in enumerate(prompts_dataloader):
                # Run a forward pass and get the activations
                with torch.no_grad():
                    _logits, cache = src_model.run_with_cache(input_ids)
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

            # Update the buffer size
            buffer_size = len(activation_buffer)

        # Whilst the buffer is less than the minimum, train the autoencoder
        train(autoencoder, activation_buffer, l1_coefficient, min_buffer)

        # Update the buffer size
        buffer_size = len(activation_buffer)
