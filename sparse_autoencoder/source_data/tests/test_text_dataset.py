"""Pile Uncopyrighted Dataset Tests."""
import pytest
from transformers import PreTrainedTokenizerFast

from sparse_autoencoder.source_data.text_dataset import GenericTextDataset


@pytest.mark.parametrize("context_size", [50, 250])
def test_tokenized_prompts_correct_size(context_size: int) -> None:
    """Test that the tokenized prompts have the correct context size."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")

    data = GenericTextDataset(tokenizer=tokenizer, context_size=context_size)

    # Check the first 100 items
    iterable = iter(data.dataset)
    for _ in range(100):
        item = next(iterable)
        assert len(item["input_ids"]) == context_size

        # Check the tokens are integers
        for token in item["input_ids"]:
            assert isinstance(token, int)


def test_dataloader_correct_size_items() -> None:
    """Test the dataloader returns the correct number & sized items."""
    batch_size = 10
    context_size = 250
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")
    data = GenericTextDataset(tokenizer=tokenizer, context_size=context_size)
    dataloader = data.get_dataloader(batch_size=batch_size)

    checks = 100
    for item in dataloader:
        checks -= 1
        if checks == 0:
            break

        tokens = item["input_ids"]
        assert tokens.shape[0] == batch_size
        assert tokens.shape[1] == context_size
