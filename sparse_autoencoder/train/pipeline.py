"""Training Pipeline."""
from collections.abc import Iterable

from jaxtyping import Int
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_store.base_store import ActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset
from sparse_autoencoder.source_data.abstract_dataset import (
    SourceDataset,
    TorchTokenizedPrompts,
)
from sparse_autoencoder.train.generate_activations import generate_activations
from sparse_autoencoder.train.resample_neurons import resample_dead_neurons
from sparse_autoencoder.train.sweep_config import SweepParametersRuntime
from sparse_autoencoder.train.train_autoencoder import train_autoencoder


def stateful_dataloader_iterable(
    dataloader: DataLoader[TorchTokenizedPrompts]
) -> Iterable[TorchTokenizedPrompts]:
    """Create a stateful dataloader iterable.

    Create an iterable that maintains it's position in the dataloader between loops.

    Examples:
        Without this, when iterating over a DataLoader with 2 loops, each loop get the same data
        (assuming shuffle is turned off). That is to say, the second loop won't maintain the
        position from where the first loop left off.

        >>> from datasets import Dataset
        >>> from torch.utils.data import DataLoader
        >>> def gen():
        ...     yield {"int": 0}
        ...     yield {"int": 1}
        >>> data = DataLoader(Dataset.from_generator(gen))
        >>> next(iter(data))["int"], next(iter(data))["int"]
        (tensor([0]), tensor([0]))

        By contrast if you create a stateful iterable from the dataloader, each loop will get
        different data.

        >>> iterator = stateful_dataloader_iterable(data)
        >>> next(iterator)["int"], next(iterator)["int"]
        (tensor([0]), tensor([1]))

    Args:
        dataloader: PyTorch DataLoader.

    Returns:
        Stateful iterable over the data in the dataloader.
    """
    yield from dataloader


def pipeline(  # noqa: PLR0913
    src_model: HookedTransformer,
    src_model_activation_hook_point: str,
    src_model_activation_layer: int,
    source_dataset: SourceDataset,
    activation_store: ActivationStore,
    num_activations_before_training: int,
    autoencoder: SparseAutoencoder,
    source_dataset_batch_size: int = 16,
    resample_frequency: int = 25_000_000,
    sweep_parameters: SweepParametersRuntime = SweepParametersRuntime(),  # noqa: B008
    device: torch.device | None = None,
    max_activations: int = 100_000_000,
) -> None:
    """Full pipeline for training the sparse autoEncoder.

    The pipeline alternates between generating activations and training the autoencoder.

    Args:
        src_model: The model to get activations from.
        src_model_activation_hook_point: The hook point to get activations from.
        src_model_activation_layer: The layer to get activations from. This is used to stop the
            model after this layer, as we don't need the final logits.
        source_dataset: Source dataset containing source model inputs (typically batches of prompts)
            that are used to generate the activations data.
        activation_store: The store to buffer activations in once generated, before training the
            autoencoder.
        num_activations_before_training: The number of activations to generate before training the
            autoencoder. As a guide, 1 million activations, each of size 1024, will take up about
            2GB of memory (assuming float16/bfloat16).
        autoencoder: The autoencoder to train.
        source_dataset_batch_size: Batch size of tokenized prompts for generating the source data.
        resample_frequency: How often to resample neurons (number of activations learnt on).
        sweep_parameters: Parameter config to use.
        device: Device to run pipeline on.
        max_activations: Maximum number of activations to train with. May train for less if the
            source dataset is exhausted.
    """
    autoencoder.to(device)

    optimizer = AdamWithReset(
        autoencoder.parameters(),
        lr=sweep_parameters.lr,
        betas=(sweep_parameters.adam_beta_1, sweep_parameters.adam_beta_2),
        eps=sweep_parameters.adam_epsilon,
        weight_decay=sweep_parameters.adam_weight_decay,
        named_parameters=autoencoder.named_parameters(),
    )

    source_dataloader = source_dataset.get_dataloader(source_dataset_batch_size)
    source_data_iterator = stateful_dataloader_iterable(source_dataloader)

    total_steps: int = 0
    activations_since_resampling: int = 0
    neuron_activity: Int[Tensor, " learned_features"] = torch.zeros(
        autoencoder.n_learned_features,
        dtype=torch.int32,
        device=device,
    )
    total_activations: int = 0

    # Run loop until source data is exhausted:
    with logging_redirect_tqdm(), tqdm(
        desc="Total activations trained on",
        dynamic_ncols=True,
        total=max_activations,
        postfix={"Current mode": "initializing"},
    ) as progress_bar:
        while total_activations < max_activations:
            # Add activations to the store
            activation_store.empty()  # In case it was filled by a different run
            progress_bar.set_postfix({"Current mode": "generating"})
            generate_activations(
                src_model,
                src_model_activation_layer,
                src_model_activation_hook_point,
                activation_store,
                source_data_iterator,
                device=device,
                context_size=source_dataset.context_size,
                num_items=num_activations_before_training,
                batch_size=source_dataset_batch_size,
            )
            if len(activation_store) == 0:
                break

            # Shuffle the store if it has a shuffle method - it is often more efficient to
            # create a shuffle method ourselves rather than get the DataLoader to shuffle
            activation_store.shuffle()

            # Train the autoencoder
            progress_bar.set_postfix({"Current mode": "training"})
            train_steps, learned_activations_fired_count = train_autoencoder(
                activation_store=activation_store,
                autoencoder=autoencoder,
                optimizer=optimizer,
                sweep_parameters=sweep_parameters,
                device=device,
                previous_steps=total_steps,
            )
            total_steps += train_steps

            if activations_since_resampling >= resample_frequency / 2:
                neuron_activity.add_(learned_activations_fired_count)

            activations_since_resampling += len(activation_store)
            total_activations += len(activation_store)
            progress_bar.update(len(activation_store))

            # Resample neurons if required
            if activations_since_resampling >= resample_frequency:
                progress_bar.set_postfix({"Current mode": "resampling"})
                activations_since_resampling = 0
                resample_dead_neurons(
                    neuron_activity=neuron_activity,
                    store=activation_store,
                    autoencoder=autoencoder,
                    sweep_parameters=sweep_parameters,
                )
                learned_activations_fired_count.zero_()
                optimizer.reset_state_all_parameters()

            activation_store.empty()
