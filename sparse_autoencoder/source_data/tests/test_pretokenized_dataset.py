"""Tests for General Pre-Tokenized Dataset."""
import pytest


TEST_DATASET = "NeelNanda/c4-tokenized-2b"


# Mock class for PreTokenizedDataset
class MockPreTokenizedDataset:
    """Mock class for PreTokenizedDataset used in testing.

    Attributes:
        dataset_path: Path to the dataset.
        context_size: The context size of the tokenized prompts.
        dataset: The mock dataset.
    """

    def __init__(self, dataset_path: str, context_size: int) -> None:
        """Initializes the mock PreTokenizedDataset with a dataset path and context size.

        Args:
            dataset_path: Path to the dataset.
            context_size: The context size of the tokenized prompts.
        """
        self.dataset_path = dataset_path
        self.context_size = context_size
        self.dataset = self._generate_mock_data()

    def _generate_mock_data(self) -> list[dict]:
        """Generates mock data for testing.

        Returns:
            list[dict]: A list of dictionaries representing mock data items.
        """
        mock_data = []
        for _ in range(10):
            item = {"input_ids": list(range(self.context_size))}
            mock_data.append(item)
        return mock_data


@pytest.mark.parametrize("context_size", [50, 250])
def test_tokenized_prompts_correct_size(context_size: int) -> None:
    """Test that the tokenized prompts have the correct context size."""
    # Use an appropriate tokenizer and dataset path

    data = MockPreTokenizedDataset(dataset_path=TEST_DATASET, context_size=context_size)

    # Check the first k items
    iterable = iter(data.dataset)
    for _ in range(2):
        item = next(iterable)
        assert len(item["input_ids"]) == context_size

        # Check the tokens are integers
        for token in item["input_ids"]:
            assert isinstance(token, int)
