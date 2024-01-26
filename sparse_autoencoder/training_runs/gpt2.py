"""Run an sweep on all layers of GPT2 Small.

Command:

```bash
git clone https://github.com/ai-safety-foundation/sparse_autoencoder.git && cd sparse_autoencoder &&
poetry env use python3.11 && poetry install &&
poetry run python sparse_autoencoder/training_runs/gpt2.py
```
"""
import os

from sparse_autoencoder import (
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
    Hyperparameters,
    LossHyperparameters,
    Method,
    OptimizerHyperparameters,
    Parameter,
    PipelineHyperparameters,
    SourceDataHyperparameters,
    SourceModelHyperparameters,
    SweepConfig,
    sweep,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train() -> None:
    """Train."""
    sweep_config = SweepConfig(
        parameters=Hyperparameters(
            loss=LossHyperparameters(
                l1_coefficient=Parameter(values=[0.0001]),
            ),
            optimizer=OptimizerHyperparameters(
                lr=Parameter(value=0.0001),
            ),
            source_model=SourceModelHyperparameters(
                name=Parameter("gpt2"),
                cache_names=Parameter(
                    value=[f"blocks.{layer}.hook_mlp_out" for layer in range(12)]
                ),
                hook_dimension=Parameter(768),
            ),
            source_data=SourceDataHyperparameters(
                dataset_path=Parameter("alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"),
                context_size=Parameter(256),
                pre_tokenized=Parameter(value=True),
                pre_download=Parameter(value=True),
                # Total dataset is c.7bn activations (64 files)
                # C. 1.5TB needed to store all activations
                dataset_files=Parameter(
                    [f"data/train-{str(i).zfill(5)}-of-00064.parquet" for i in range(20)]
                ),
            ),
            autoencoder=AutoencoderHyperparameters(expansion_factor=Parameter(values=[32, 64])),
            pipeline=PipelineHyperparameters(),
            activation_resampler=ActivationResamplerHyperparameters(
                threshold_is_dead_portion_fires=Parameter(1e-5),
            ),
        ),
        method=Method.GRID,
    )

    sweep(sweep_config=sweep_config)


if __name__ == "__main__":
    train()
