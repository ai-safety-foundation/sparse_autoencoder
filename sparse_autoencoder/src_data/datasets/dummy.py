"""Dummy dataset for testing/examples."""
from jaxtyping import Int
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sparse_autoencoder.src_data.src_data import CollateResponseTokens


class RandomIntDataset(Dataset[Int[Tensor, " pos"]]):
    """Dummy dataset for testing/examples."""

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        pos: int,
        vocab_size: int = 50000,
    ):
        """Initialise the dataset.

        Args:
            num_samples: Number of items in the dataset.
            batch_size: Batch size.
            pos: Number of tokens in each item.
            vocab_size: Size of the vocabulary.
        """
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.pos = pos
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        """Length Dunder Method."""
        return self.num_samples

    def __getitem__(self, _idx: int) -> Int[Tensor, " pos"]:
        """Get Item Dunder Method.

        Args:
            idx: Index of the item to get.
        """
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.pos,),
            dtype=torch.long,
        )


def dummy_collate_fn(
    batch: list[Tensor],
) -> CollateResponseTokens:
    """Dummy Collate Fn."""
    return torch.stack(batch)


def create_dummy_dataloader(
    num_samples: int,
    batch_size: int,
    pos: int = 512,
    vocab_size: int = 50000,
) -> DataLoader[Int[Tensor, " pos"]]:
    """Create dummy dataloader."""
    dataset = RandomIntDataset(num_samples, batch_size, pos, vocab_size)
    return DataLoader(dataset, collate_fn=dummy_collate_fn)
