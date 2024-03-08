"""Sweep."""
from pathlib import Path
import re
import sys
import traceback

import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import wandb

from sparse_autoencoder.autoencoder.lightning import (
    LitSparseAutoencoder,
    LitSparseAutoencoderConfig,
)
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset
from sparse_autoencoder.source_data.pretokenized_dataset import PreTokenizedDataset
from sparse_autoencoder.source_data.text_dataset import TextDataset
from sparse_autoencoder.train.pipeline import Pipeline
from sparse_autoencoder.train.sweep_config import (
    RuntimeHyperparameters,
    SweepConfig,
)
from sparse_autoencoder.utils.data_parallel import DataParallelWithModelAttributes


def setup_source_model(
    hyperparameters: RuntimeHyperparameters,
) -> HookedTransformer | DataParallelWithModelAttributes[HookedTransformer]:
    """Setup the source model using HookedTransformer.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        The initialized source model.
    """
    model = HookedTransformer.from_pretrained(
        hyperparameters["source_model"]["name"],
        dtype=hyperparameters["source_model"]["dtype"],
    )

    return DataParallelWithModelAttributes(model)


def setup_autoencoder(
    hyperparameters: RuntimeHyperparameters,
) -> LitSparseAutoencoder:
    """Setup the sparse autoencoder.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        The initialized sparse autoencoder.
    """
    autoencoder_input_dim: int = hyperparameters["source_model"]["hook_dimension"]
    expansion_factor = hyperparameters["autoencoder"]["expansion_factor"]

    config = LitSparseAutoencoderConfig(
        n_input_features=autoencoder_input_dim,
        n_learned_features=autoencoder_input_dim * expansion_factor,
        n_components=len(hyperparameters["source_model"]["cache_names"]),
        component_names=hyperparameters["source_model"]["cache_names"],
        l1_coefficient=hyperparameters["loss"]["l1_coefficient"],
        resample_interval=hyperparameters["activation_resampler"]["resample_interval"],
        max_n_resamples=hyperparameters["activation_resampler"]["max_n_resamples"],
        resample_dead_neurons_dataset_size=hyperparameters["activation_resampler"][
            "n_activations_activity_collate"
        ],
        resample_loss_dataset_size=hyperparameters["activation_resampler"]["resample_dataset_size"],
        resample_threshold_is_dead_portion_fires=hyperparameters["activation_resampler"][
            "threshold_is_dead_portion_fires"
        ],
    )

    return LitSparseAutoencoder(config)


def setup_source_data(hyperparameters: RuntimeHyperparameters) -> SourceDataset:
    """Setup the source data for training.

    Args:
        hyperparameters: The hyperparameters dictionary.

    Returns:
        The initialized source dataset.

    Raises:
        ValueError: If the tokenizer name is not specified, but pre_tokenized is False.
    """
    dataset_dir = (
        hyperparameters["source_data"]["dataset_dir"]
        if "dataset_dir" in hyperparameters["source_data"]
        else None
    )

    dataset_files = (
        hyperparameters["source_data"]["dataset_files"]
        if "dataset_files" in hyperparameters["source_data"]
        else None
    )

    if hyperparameters["source_data"]["pre_tokenized"]:
        return PreTokenizedDataset(
            context_size=hyperparameters["source_data"]["context_size"],
            dataset_dir=dataset_dir,
            dataset_files=dataset_files,
            dataset_path=hyperparameters["source_data"]["dataset_path"],
            dataset_column_name=hyperparameters["source_data"]["dataset_column_name"],
            pre_download=hyperparameters["source_data"]["pre_download"],
        )

    if hyperparameters["source_data"]["tokenizer_name"] is None:
        error_message = (
            "If pre_tokenized is False, then tokenizer_name must be specified in the "
            "hyperparameters."
        )
        raise ValueError(error_message)

    tokenizer = AutoTokenizer.from_pretrained(hyperparameters["source_data"]["tokenizer_name"])

    return TextDataset(
        context_size=hyperparameters["source_data"]["context_size"],
        dataset_column_name=hyperparameters["source_data"]["dataset_column_name"],
        dataset_dir=dataset_dir,
        dataset_files=dataset_files,
        dataset_path=hyperparameters["source_data"]["dataset_path"],
        n_processes_preprocessing=4,
        pre_download=hyperparameters["source_data"]["pre_download"],
        tokenizer=tokenizer,
    )


def setup_wandb() -> RuntimeHyperparameters:
    """Initialise wandb for experiment tracking."""
    wandb.run = None  # Fix for broken pipe bug in wandb
    wandb.init()
    return dict(wandb.config)  # type: ignore


def stop_layer_from_cache_names(cache_names: list[str]) -> int:
    """Get the stop layer from the cache names.

    Examples:
        >>> cache_names = [
        ...     "blocks.0.hook_mlp_out",
        ...     "blocks.1.hook_mlp_out",
        ...     "blocks.2.hook_mlp_out",
        ...     ]
        >>> stop_layer_from_cache_names(cache_names)
        2

        >>> cache_names = [
        ...     "blocks.0.hook_x.0.y",
        ...     "blocks.0.hook_x.1.y",
        ...     ]
        >>> stop_layer_from_cache_names(cache_names)
        0

    Args:
        cache_names: The cache names.

    Returns:
        The stop layer.

    Raises:
        ValueError: If no number is found in the cache names.
    """
    cache_layers: list[int] = []

    first_n_in_string_regex = re.compile(r"[0-9]+")

    for cache_name in cache_names:
        cache_layer = first_n_in_string_regex.findall(cache_name)
        if len(cache_layer) == 0:
            error_message = f"Could not find a number in the cache name {cache_name}."
            raise ValueError(error_message)
        cache_layers.append(int(cache_layer[0]))

    return max(cache_layers)


def run_training_pipeline(
    hyperparameters: RuntimeHyperparameters,
    source_model: HookedTransformer | DataParallelWithModelAttributes[HookedTransformer],
    autoencoder: LitSparseAutoencoder,
    source_data: SourceDataset,
    run_name: str,
) -> None:
    """Run the training pipeline for the sparse autoencoder.

    Args:
        hyperparameters: The hyperparameters dictionary.
        source_model: The source model.
        autoencoder: The sparse autoencoder.
        source_data: The source data.
        run_name: The name of the run.
    """
    checkpoint_path = Path(__file__).parent.parent.parent / ".checkpoints"
    checkpoint_path.mkdir(exist_ok=True)

    random_seed = hyperparameters["random_seed"]
    torch.random.manual_seed(random_seed)

    cache_names = hyperparameters["source_model"]["cache_names"]
    stop_layer = stop_layer_from_cache_names(cache_names)

    pipeline = Pipeline(
        autoencoder=autoencoder,
        cache_names=cache_names,
        checkpoint_directory=checkpoint_path,
        layer=stop_layer,
        source_data_batch_size=hyperparameters["pipeline"]["source_data_batch_size"],
        source_dataset=source_data,
        source_model=source_model,
        log_frequency=hyperparameters["pipeline"]["log_frequency"],
        run_name=run_name,
        num_workers_data_loading=hyperparameters["pipeline"]["num_workers_data_loading"],
        n_input_features=hyperparameters["source_model"]["hook_dimension"],
        n_learned_features=int(
            hyperparameters["autoencoder"]["expansion_factor"]
            * hyperparameters["source_model"]["hook_dimension"]
        ),
    )

    pipeline.run_pipeline(
        train_batch_size=hyperparameters["pipeline"]["train_batch_size"],
        max_store_size=hyperparameters["pipeline"]["max_store_size"],
        max_activations=hyperparameters["pipeline"]["max_activations"],
        checkpoint_frequency=hyperparameters["pipeline"]["checkpoint_frequency"],
        validate_frequency=hyperparameters["pipeline"]["validation_frequency"],
        validation_n_activations=hyperparameters["pipeline"]["validation_n_activations"],
    )


def train() -> None:
    """Train the sparse autoencoder using the hyperparameters from the WandB sweep."""
    try:
        # Set up WandB
        hyperparameters = setup_wandb()
        run_name: str = wandb.run.name  # type: ignore

        # Set up the source model
        source_model = setup_source_model(hyperparameters)

        # Set up the autoencoder, optimizer and learning rate scheduler
        autoencoder = setup_autoencoder(hyperparameters)

        # Set up the source data
        source_data = setup_source_data(hyperparameters)

        # Run the training pipeline
        run_training_pipeline(
            hyperparameters=hyperparameters,
            source_model=source_model,
            autoencoder=autoencoder,
            source_data=source_data,
            run_name=run_name,
        )
    except Exception as _e:  # noqa: BLE001
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)  # noqa: T201
        sys.exit(1)


def sweep(
    sweep_config: SweepConfig | None = None,
    sweep_id: str | None = None,
) -> None:
    """Run the training pipeline with wandb hyperparameter sweep.

    Warning:
        Either sweep_config or sweep_id must be specified, but not both.

    Args:
        sweep_config: The sweep configuration.
        sweep_id: The sweep id for an existing sweep.

    Raises:
        ValueError: If neither sweep_config nor sweep_id is specified.
    """
    if sweep_id is not None:
        wandb.agent(sweep_id, train, project="sparse-autoencoder")

    elif sweep_config is not None:
        sweep_id = wandb.sweep(sweep_config.to_dict(), project="sparse-autoencoder")
        wandb.agent(sweep_id, train)

    else:
        error_message = "Either sweep_config or sweep_id must be specified."
        raise ValueError(error_message)

    wandb.finish()
