"""Pile Uncopyrighted Dataset Tests."""
import pytest
from transformers import PreTrainedTokenizerFast

from sparse_autoencoder.source_data.pile_uncopyrighted import PileUncopyrightedDataset


@pytest.mark.parametrize("context_size", [50, 250])
def test_tokenized_prompts_correct_size(context_size: int) -> None:
    """Test that the tokenized prompts have the correct context size."""
    tokenizer = PreTrainedTokenizerFast.from_pretrained("gpt2")

    data = PileUncopyrightedDataset(tokenizer=tokenizer, context_size=context_size)

    # Check the first 100 items
    iterable = iter(data.dataset)
    for _ in range(100):
        item = next(iterable)
        assert len(item["input_ids"]) == context_size

        # Check the tokens are integers
        for token in item["input_ids"]:
            assert isinstance(token, int)
