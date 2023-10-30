"""Activation Buffer."""
import random
from enum import StrEnum

import torch
import torch.utils.data.dataloader
from jaxtyping import Float
from torch import Tensor


class StorageType(StrEnum):
    """Storage Type."""

    GPU = "GPU"
    # RAM = "RAM"
    # DISK = "DISK"
    # NETWORK = "NETWORK"


class ActivationBuffer:
    """Single Threaded Activation Buffer."""

    def __len__(self) -> int:
        return len(self.gpu_tensors)

    gpu_tensors: list[Float[Tensor, "input_activations"]] = []

    storage_type: StorageType

    def __init__(self, storage_type: StorageType = StorageType.GPU) -> None:
        self.storage_type = storage_type

    def append(self, batch: Float[Tensor, "batch input_activations"]) -> None:
        """Add a batch to the buffer"""
        list_tensors = torch.unbind(batch)
        self.gpu_tensors.extend(list_tensors)

    def sample_without_replace(
        self, batch_size: int, device: torch.device | None = None
    ) -> Float[Tensor, "batch input_activations"]:
        """Sample a batch from the buffer (and remove it)"""
        sample = []

        for _batch_idx in range(batch_size):
            rand_idx = random.randint(0, len(self))
            sample.append(self.gpu_tensors[rand_idx])
            self.gpu_tensors[rand_idx] = self.gpu_tensors[-1]
            self.gpu_tensors.pop()

        return torch.stack(sample).to(device)

    def sample_with_replace(
        self, batch_size: int, device: torch.device | None = None
    ) -> Float[Tensor, "batch input_activations"]:
        """Sample a batch from the buffer (and remove it)"""
        sample = []

        for _batch_idx in range(batch_size):
            rand_idx = random.randint(0, len(self))
            sample.append(self.gpu_tensors[rand_idx])

        return torch.stack(sample).to(device)
