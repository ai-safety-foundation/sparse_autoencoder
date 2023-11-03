"""Training Pipeline."""
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer

import wandb
from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.train.generate_activations import generate_activations
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime
from sparse_autoencoder.train.train_autoencoder import train_autoencoder


def pipeline(
    src_model: HookedTransformer,
    src_model_activation_hook_point: str,
    src_model_activation_layer: int,
    src_dataloader: DataLoader,
    activation_store: ActivationStore,
    num_activations_before_training: int,
    autoencoder: SparseAutoencoder,
    sweep_parameters: SweepParametersRuntime = SweepParametersRuntime(),
):
    """Full pipeline for training the sparse autoEncoder.

    Args:
        src_model: The model to get activations from.
        src_model_activation_hook_point: The hook point to get activations from.
        src_model_activation_layer: The layer to get activations from. This is used to stop the
            model after this layer, as we don't need the final logits.
        src_dataloader: DataLoader containing source model inputs (typically batches of prompts)
            that are used to generate the activations data.
        activation_store: The store to buffer activations in once generated, before training the
            autoencoder.
        num_activations_before_training: The number of activations to generate before training the
            autoencoder. Once this number is generated, training will start. Once the date is
            exhausted, training will pause and the store will be filled again up to this number.
            This is repeated until the src_dataloader is exhausted.
        autoencoder: The autoencoder to train.
        sweep_parameters: Parameter config to use.
    """
    # Initialise wandb sweep
    wandb.init(project="sparse-autoencoder", config=sweep_parameters)

    # Get hyperparameters
    optimizer = Adam(
        autoencoder.parameters(),
        lr=sweep_parameters.lr,
        betas=(sweep_parameters.adam_beta_1, sweep_parameters.adam_beta_2),
        eps=sweep_parameters.adam_epsilon,
        weight_decay=sweep_parameters.adam_weight_decay,
    )

    # Run loop until data is exhausted:
    with tqdm(desc="Number of generate-train loops") as progress_bar:
        while True:
            progress_bar.update(1)

            # Add activations to the store
            generate_activations(
                src_model,
                src_model_activation_layer,
                src_model_activation_hook_point,
                activation_store,
                src_dataloader,
                num_activations_before_training,
            )
            if len(activation_store) == 0:
                break

            # Shuffle the store if it has a shuffle method - it is often more efficient to create a
            # shuffle method ourselves rather than get the DataLoader to shuffle
            if hasattr(activation_store, "shuffle"):
                activation_store.shuffle()

            # Create a dataloader from the store
            dataloader = DataLoader(
                activation_store, batch_size=sweep_parameters.batch_size
            )

            # Train the autoencoder
            train_autoencoder(dataloader, autoencoder, optimizer, sweep_parameters)

            # Empty the store so we can fill it up again
            activation_store.empty()
