"""Tests for General Pre-Tokenized Dataset."""
import pytest

from sparse_autoencoder.source_data.pretokenized_dataset import PreTokenizedDataset


TEST_DATASET = "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"


@pytest.mark.integration_test()
@pytest.mark.parametrize("context_size", [128, 256])
def test_tokenized_prompts_correct_size(context_size: int) -> None:
    """Test that the tokenized prompts have the correct context size."""
    data = PreTokenizedDataset(dataset_path=TEST_DATASET, context_size=context_size)

    # Check the first k items
    iterable = iter(data.dataset)
    for _ in range(2):
        item = next(iterable)
        assert len(item["input_ids"]) == context_size

        # Check the tokens are integers
        for token in item["input_ids"]:
            assert isinstance(token, int)


@pytest.mark.integration_test()
def test_fails_context_size_too_large() -> None:
    """Test that it fails if the context size is set as larger than the source dataset on HF."""
    data = PreTokenizedDataset(dataset_path=TEST_DATASET, context_size=512)
    with pytest.raises(ValueError, match=r"larger than the tokenized prompt size"):
        next(iter(data))
