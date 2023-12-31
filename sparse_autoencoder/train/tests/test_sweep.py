"""Tests for sweep functionality."""

import pytest
from syrupy.session import SnapshotSession
import torch

from sparse_autoencoder.train.sweep import (
    setup_activation_resampler,
    setup_autoencoder,
    setup_loss_function,
)
from sparse_autoencoder.train.sweep_config import (
    RuntimeHyperparameters,
)


@pytest.fixture()
def dummy_hyperparameters() -> RuntimeHyperparameters:
    """Sweep config dummy fixture."""
    return {
        "activation_resampler": {
            "threshold_is_dead_portion_fires": 0.0,
            "max_n_resamples": 4,
            "n_activations_activity_collate": 100_000_000,
            "resample_dataset_size": 819_200,
            "resample_interval": 200_000_000,
        },
        "autoencoder": {"expansion_factor": 4},
        "loss": {"l1_coefficient": 0.0001},
        "optimizer": {
            "adam_beta_1": 0.9,
            "adam_beta_2": 0.99,
            "adam_weight_decay": 0,
            "amsgrad": False,
            "fused": False,
            "lr": 1e-05,
        },
        "pipeline": {
            "checkpoint_frequency": 100000000,
            "log_frequency": 100,
            "max_activations": 2000000000,
            "max_store_size": 3145728,
            "source_data_batch_size": 12,
            "train_batch_size": 4096,
            "validation_frequency": 314572800,
            "validation_n_activations": 1024,
        },
        "random_seed": 49,
        "source_data": {
            "context_size": 128,
            "dataset_column_name": "input_ids",
            "dataset_dir": None,
            "dataset_files": None,
            "dataset_path": "NeelNanda/c4-code-tokenized-2b",
            "pre_download": False,
            "pre_tokenized": True,
            "tokenizer_name": None,
        },
        "source_model": {
            "dtype": "float32",
            "hook_dimension": 512,
            "cache_names": ["mlp_out"],
            "name": "gelu-2l",
        },
    }


def test_setup_activation_resampler(
    dummy_hyperparameters: RuntimeHyperparameters, snapshot: SnapshotSession
) -> None:
    """Test the setup_activation_resampler function."""
    activation_resampler = setup_activation_resampler(dummy_hyperparameters)
    assert snapshot == str(
        activation_resampler
    ), "Activation resampler string representation has changed."


def test_setup_autoencoder(
    dummy_hyperparameters: RuntimeHyperparameters, snapshot: SnapshotSession
) -> None:
    """Test the setup_autoencoder function."""
    autoencoder = setup_autoencoder(dummy_hyperparameters, device=torch.device("cpu"))
    assert snapshot == str(autoencoder), "Autoencoder string representation has changed."


def test_setup_loss_function(
    dummy_hyperparameters: RuntimeHyperparameters, snapshot: SnapshotSession
) -> None:
    """Test the setup_loss_function function."""
    loss_function = setup_loss_function(dummy_hyperparameters)
    assert snapshot == str(loss_function), "Loss function string representation has changed."
