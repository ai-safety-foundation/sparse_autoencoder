"""Activation Buffer."""
import random

import torch
import torch.utils.data.dataloader
from jaxtyping import Float
from torch import Tensor


class ActivationBuffer:
    """Single Threaded Activation Buffer."""

    def __len__(self) -> int:
        return len(self.tensor_pointers)

    tensor_pointers: list[Float[Tensor, "input_activations"]] = []

    def append(self, batch: Float[Tensor, "batch input_activations"]) -> None:
        """Add a batch to the buffer"""
        list_tensors = torch.unbind(batch)
        self.tensor_pointers.extend(list_tensors)

    def sample_without_replace(
        self, batch_size: int, device: torch.device | None = None
    ) -> Float[Tensor, "batch input_activations"]:
        """Sample a batch from the buffer (and remove it)"""
        sample = []

        for _batch_idx in range(batch_size):
            rand_idx = random.randint(0, len(self))
            sample.append(self.tensor_pointers[rand_idx])
            self.tensor_pointers[rand_idx] = self.tensor_pointers[-1]
            self.tensor_pointers.pop()

        return torch.stack(sample).to(device)

    def sample_with_replace(
        self, batch_size: int, device: torch.device | None = None
    ) -> Float[Tensor, "batch input_activations"]:
        """Sample a batch from the buffer (and remove it)"""
        sample = []

        for _batch_idx in range(batch_size):
            rand_idx = random.randint(0, len(self))
            sample.append(self.tensor_pointers[rand_idx])

        return torch.stack(sample).to(device)
