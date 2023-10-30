"""Activation Buffer."""
import asyncio
import io
import random
import shutil
import threading
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import aiofiles
import torch
import torch.utils.data.dataloader
from datasets import Dataset
from jaxtyping import Float
from torch import Tensor


class ActivationStore(ABC):
    """Activation Store Abstract Class."""

    @abstractmethod
    def append(self, batch: Float[Tensor, "batch input_activations"]) -> None:
        """Add a batch to the store"""
        raise NotImplementedError

    @abstractmethod
    def sample_without_replace(
        self, batch_size: int, device: torch.device | None = None
    ) -> Float[Tensor, "batch input_activations"]:
        """Sample a batch from the store (and remove it)"""
        raise NotImplementedError

    @abstractmethod
    def sample_with_replace(
        self, batch_size: int, device: torch.device | None = None
    ) -> Float[Tensor, "batch input_activations"]:
        """Sample a batch from the store (and remove it)"""
        raise NotImplementedError


class DiskActivationStore(ActivationStore):
    """Disk Activation Store."""

    storage_dir: Path
    _cache: list[Float[Tensor, "batch input_activations"]]
    max_batches_cache: int

    def __init__(
        self, storage_dir: Path, max_batches_cache: int, empty_dir: bool = True
    ) -> None:
        self.storage_dir = storage_dir
        self._cache = []
        self.max_batches_cache = max_batches_cache

        # Delete the directory if empty_dir is True
        if empty_dir:
            shutil.rmtree(self.storage_dir, ignore_errors=True)

        # Create the directory if it doesn't exist
        storage_dir.mkdir(parents=True, exist_ok=True)

    def append(self, batch: Float[Tensor, "batch input_activations"]) -> None:
        """Add a batch to the store."""
        self._cache.append(batch)

        if len(self._cache) >= self.max_batches_cache:
            self._write_cache_to_disk()

    def _write_cache_to_disk(self):
        """Write the current cache to the disk."""
        # Capture the current cache for writing
        batches_to_write = self._cache
        self._cache = []

        # Use a background thread to run the async method
        threading.Thread(target=self._async_write, args=(batches_to_write,)).start()

    async def _async_write(
        self, batches: list[Float[Tensor, "batch input_activations"]]
    ):
        """Asynchronous helper method to write batches to disk."""
        filename = f"{uuid.uuid4().hex}.pt"

        # Combine all batches for serialization
        combined_batch = torch.cat(batches, dim=0)

        async with aiofiles.open(self.storage_dir / filename, mode="wb") as f:
            buffer = io.BytesIO()
            torch.save(combined_batch, buffer)

        await f.write(buffer)


class ListPointerGPUBuffer(ActivationStore):
    """Single Threaded List Pointer GPU Buffer."""

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
            rand_idx = random.randint(0, len(self.tensor_pointers) - 1)
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
