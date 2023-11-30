"""Test the abstract dataset."""

import pytest

from sparse_autoencoder.source_data.abstract_dataset import SourceDataset
from sparse_autoencoder.source_data.mock_dataset import MockDataset


@pytest.fixture()
def mock_dataset() -> MockDataset:
    """Fixture to create a default ConsecutiveIntHuggingFaceDataset for testing.

    Returns:
        ConsecutiveIntHuggingFaceDataset: An instance of the dataset for testing.
    """
    return MockDataset(context_size=10, buffer_size=100)


def test_extended_dataset_initialization(mock_dataset: MockDataset) -> None:
    """Test the initialization of the extended dataset."""
    assert mock_dataset is not None
    assert isinstance(mock_dataset, SourceDataset)


def test_extended_dataset_iterator(mock_dataset: MockDataset) -> None:
    """Test the iterator of the extended dataset."""
    iterator = iter(mock_dataset)
    assert iterator is not None


def test_get_dataloader(mock_dataset: MockDataset) -> None:
    """Test the get_dataloader method of the extended dataset."""
    batch_size = 3
    dataloader = mock_dataset.get_dataloader(batch_size=batch_size)
    first_item = next(iter(dataloader))["input_ids"]
    assert first_item.shape[0] == batch_size
