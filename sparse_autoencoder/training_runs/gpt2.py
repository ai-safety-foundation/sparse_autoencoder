"""Run an sweep on all layers of GPT2 Small.

Command:

```bash
git clone https://github.com/ai-safety-foundation/sparse_autoencoder.git && cd sparse_autoencoder &&
poetry env use python3.11 && poetry install &&
poetry run python sparse_autoencoder/training_runs/gpt2.py
```
"""
import os

from lightning import Trainer

from sparse_autoencoder.autoencoder.lightning import (
    LitSparseAutoencoder,
    LitSparseAutoencoderConfig,
)
from sparse_autoencoder.source_data.pretokenized_dataset import PreTokenizedDataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train() -> None:
    """Train."""
    # Also set resampler threshold dead to 1e-5?
    config = LitSparseAutoencoderConfig(
        source_model_name="gpt2",
        component_names=[f"blocks.{layer}.hook_mlp_out" for layer in range(12)],
        l1_coefficient=0.0001,
        learning_rate=0.0001,
        n_input_features=768,
        n_learned_features=32 * 768,
        n_components=12,
    )

    model = LitSparseAutoencoder(config)

    trainer = Trainer()

    dataset = PreTokenizedDataset(
        dataset_path="alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2",
        dataset_files=[f"data/train-{str(i).zfill(5)}-of-00064.parquet" for i in range(1)],
        pre_download=True,
    )

    dataloader = dataset.get_dataloader(batch_size=32, num_workers=2)

    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    train()
