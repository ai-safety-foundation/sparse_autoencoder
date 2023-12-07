"""Sweep."""
from pathlib import Path

import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name, get_device
import wandb

from sparse_autoencoder import (
    ActivationResampler,
    AdamWithReset,
    L2ReconstructionLoss,
    LearnedActivationsL1Loss,
    LossReducer,
    Pipeline,
    PreTokenizedDataset,
    SparseAutoencoder,
)
from sparse_autoencoder.train.sweep_config import (
    RuntimeHyperparameters,
    SweepConfig,
)


def setup_activation_resampler(hyperparameters: RuntimeHyperparameters) -> ActivationResampler:
    """Setup the activation resampler for the autoencoder.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        ActivationResampler: The initialized activation resampler.
    """
    return ActivationResampler(
        resample_interval=hyperparameters["activation_resampler"]["resample_interval"],
        max_resamples=hyperparameters["activation_resampler"]["max_resamples"],
        n_steps_collate=hyperparameters["activation_resampler"]["n_steps_collate"],
        resample_dataset_size=hyperparameters["activation_resampler"]["resample_dataset_size"],
        dead_neuron_threshold=hyperparameters["activation_resampler"]["dead_neuron_threshold"],
    )


def setup_source_model(hyperparameters: RuntimeHyperparameters) -> HookedTransformer:
    """Setup the source model using HookedTransformer.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        The initialized source model.
    """
    return HookedTransformer.from_pretrained(
        hyperparameters["source_model"]["name"],
        dtype=hyperparameters["source_model"]["dtype"],
    )


def setup_autoencoder(
    hyperparameters: RuntimeHyperparameters, device: torch.device
) -> SparseAutoencoder:
    """Setup the sparse autoencoder.

    Args:
        hyperparameters: The hyperparameters dictionary.
        device: The computation device.

    Returns:
        The initialized sparse autoencoder.
    """
    autoencoder_input_dim: int = hyperparameters["source_model"]["hook_dimension"]
    expansion_factor = hyperparameters["autoencoder"]["expansion_factor"]
    return SparseAutoencoder(
        n_input_features=autoencoder_input_dim,
        n_learned_features=autoencoder_input_dim * expansion_factor,
    ).to(device)


def setup_loss_function(hyperparameters: RuntimeHyperparameters) -> LossReducer:
    """Setup the loss function for the autoencoder.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        The combined loss function.
    """
    return LossReducer(
        LearnedActivationsL1Loss(
            l1_coefficient=hyperparameters["loss"]["l1_coefficient"],
        ),
        L2ReconstructionLoss(),
    )


def setup_optimizer(
    autoencoder: SparseAutoencoder, hyperparameters: RuntimeHyperparameters
) -> AdamWithReset:
    """Setup the optimizer for the autoencoder.

    Args:
        autoencoder: The sparse autoencoder model.
        hyperparameters: The hyperparameters dictionary.

    Returns:
        The initialized optimizer.
    """
    return AdamWithReset(
        params=autoencoder.parameters(),
        named_parameters=autoencoder.named_parameters(),
        lr=hyperparameters["optimizer"]["lr"],
        betas=(
            hyperparameters["optimizer"]["adam_beta_1"],
            hyperparameters["optimizer"]["adam_beta_2"],
        ),
        weight_decay=hyperparameters["optimizer"]["adam_weight_decay"],
        amsgrad=hyperparameters["optimizer"]["amsgrad"],
        fused=hyperparameters["optimizer"]["fused"],
    )


def setup_source_data(hyperparameters: RuntimeHyperparameters) -> PreTokenizedDataset:
    """Setup the source data for training.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        PreTokenizedDataset: The initialized source data.
    """
    return PreTokenizedDataset(
        dataset_path=hyperparameters["source_data"]["dataset_path"],
        context_size=hyperparameters["source_data"]["context_size"],
    )


def setup_wandb() -> RuntimeHyperparameters:
    """Initialise wandb for experiment tracking."""
    wandb.init(project="sparse-autoencoder")
    return dict(wandb.config)  # type: ignore


def run_training_pipeline(
    hyperparameters: RuntimeHyperparameters,
    source_model: HookedTransformer,
    autoencoder: SparseAutoencoder,
    loss: LossReducer,
    optimizer: AdamWithReset,
    activation_resampler: ActivationResampler,
    source_data: PreTokenizedDataset,
) -> None:
    """Run the training pipeline for the sparse autoencoder.

    Args:
        hyperparameters: The hyperparameters dictionary.
        source_model: The source model.
        autoencoder: The sparse autoencoder.
        loss: The loss function.
        optimizer: The optimizer.
        activation_resampler: The activation resampler.
        source_data: The source data.
    """
    checkpoint_path = Path("../../.checkpoints")
    checkpoint_path.mkdir(exist_ok=True)

    random_seed = hyperparameters["random_seed"]
    torch.random.manual_seed(random_seed)

    hook_point = get_act_name(
        hyperparameters["source_model"]["hook_site"], hyperparameters["source_model"]["hook_layer"]
    )
    pipeline = Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=autoencoder,
        cache_name=hook_point,
        checkpoint_directory=checkpoint_path,
        layer=hyperparameters["source_model"]["hook_layer"],
        loss=loss,
        optimizer=optimizer,
        source_data_batch_size=hyperparameters["pipeline"]["source_data_batch_size"],
        source_dataset=source_data,
        source_model=source_model,
        log_frequency=hyperparameters["pipeline"]["log_frequency"],
    )

    pipeline.run_pipeline(
        train_batch_size=hyperparameters["pipeline"]["train_batch_size"],
        max_store_size=hyperparameters["pipeline"]["max_store_size"],
        max_activations=hyperparameters["pipeline"]["max_activations"],
        checkpoint_frequency=hyperparameters["pipeline"]["checkpoint_frequency"],
        validate_frequency=hyperparameters["pipeline"]["validation_frequency"],
        validation_number_activations=hyperparameters["pipeline"]["validation_number_activations"],
    )


def sweep(sweep_config: SweepConfig) -> None:
    """Main function to run the training pipeline with wandb hyperparameter sweep."""
    sweep_id = wandb.sweep(sweep_config.to_dict(), project="sparse-autoencoder")

    def train() -> None:
        """Train the sparse autoencoder using the hyperparameters from the WandB sweep."""
        # Set up WandB
        hyperparameters = setup_wandb()

        # Setup the device for training
        device = get_device()

        # Set up the source model
        source_model = setup_source_model(hyperparameters)

        # Set up the autoencoder
        autoencoder = setup_autoencoder(hyperparameters, device)

        # Set up the loss function
        loss_function = setup_loss_function(hyperparameters)

        # Set up the optimizer
        optimizer = setup_optimizer(autoencoder, hyperparameters)

        # Set up the activation resampler
        activation_resampler = setup_activation_resampler(hyperparameters)

        # Set up the source data
        source_data = setup_source_data(hyperparameters)

        # Run the training pipeline
        run_training_pipeline(
            hyperparameters=hyperparameters,
            source_model=source_model,
            autoencoder=autoencoder,
            loss=loss_function,
            optimizer=optimizer,
            activation_resampler=activation_resampler,
            source_data=source_data,
        )

    wandb.agent(sweep_id, train)
