"""Tests for the mock dataset."""
import pytest
import torch
from torch import Tensor

from sparse_autoencoder.source_data.mock_dataset import ConsecutiveIntHuggingFaceDataset


class TestConsecutiveIntHuggingFaceDataset:
    """Tests for the ConsecutiveIntHuggingFaceDataset."""

    @pytest.fixture(scope="class")
    def create_dataset(self) -> ConsecutiveIntHuggingFaceDataset:
        """Fixture to create a default ConsecutiveIntHuggingFaceDataset for testing.

        Returns:
            ConsecutiveIntHuggingFaceDataset: An instance of the dataset for testing.
        """
        return ConsecutiveIntHuggingFaceDataset(context_size=10, vocab_size=1000, n_items=100)

    def test_dataset_initialization_failure(self) -> None:
        """Test invalid initialization failure of the ConsecutiveIntHuggingFaceDataset."""
        with pytest.raises(
            ValueError,
            match=r"n_items \(\d+\) \+ context_size \(\d+\) must be less than vocab_size \(\d+\)",
        ):
            ConsecutiveIntHuggingFaceDataset(context_size=40, vocab_size=50, n_items=20)

    def test_dataset_len(self, create_dataset: ConsecutiveIntHuggingFaceDataset) -> None:
        """Test the __len__ method of the dataset.

        Args:
            create_dataset: Fixture to create a test dataset instance.
        """
        expected_length = 100
        assert len(create_dataset) == expected_length, "Dataset length is not as expected."

    def test_dataset_getitem(self, create_dataset: ConsecutiveIntHuggingFaceDataset) -> None:
        """Test the __getitem__ method of the dataset.

        Args:
            create_dataset: Fixture to create a test dataset instance.
        """
        item = create_dataset[0]
        assert isinstance(item, dict), "Item should be a dictionary."
        assert "input_ids" in item, "Item should have 'input_ids' key."
        assert isinstance(item["input_ids"], list), "input_ids should be a list."

    def test_create_data(self, create_dataset: ConsecutiveIntHuggingFaceDataset) -> None:
        """Test the create_data method of the dataset.

        Args:
            create_dataset: Fixture to create a test dataset instance.
        """
        data: Tensor = create_dataset.create_data(n_items=10, context_size=5)
        assert data.shape == (10, 5), "Data shape is not as expected."

    def test_dataset_iteration(self, create_dataset: ConsecutiveIntHuggingFaceDataset) -> None:
        """Test the iteration functionality of the dataset.

        Args:
            create_dataset: Fixture to create a test dataset instance.
        """
        items = [item["input_ids"] for item in create_dataset]

        # Check they are all unique
        items_tensor = torch.tensor(items)
        unique_items = torch.unique(items_tensor, dim=0)
        assert items_tensor.shape == unique_items.shape, "Items are not unique."
