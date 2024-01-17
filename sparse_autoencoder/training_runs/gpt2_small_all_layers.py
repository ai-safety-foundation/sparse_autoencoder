"""Run an sweep on all layers of GPT2 Small.

Command:

```bash
git clone https://github.com/ai-safety-foundation/sparse_autoencoder.git && cd sparse_autoencoder &&
poetry env use python3.11 && poetry install &&
poetry run python sparse_autoencoder/training_runs/gpt_small_mlp_l0.py
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
)
from sparse_autoencoder import Parameter as WandbParameter
from sparse_autoencoder import (
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
                l1_coefficient=WandbParameter(values=[0.0001, 0.0005]),
            ),
            optimizer=OptimizerHyperparameters(
                lr=WandbParameter(value=0.0001),
            ),
            source_model=SourceModelHyperparameters(
                name=WandbParameter("gpt2"),
                cache_names=WandbParameter(
                    value=[f"blocks.{layer}.hook_mlp_out" for layer in range(12)]
                ),
                hook_dimension=WandbParameter(768),
            ),
            source_data=SourceDataHyperparameters(
                dataset_path=WandbParameter(
                    "alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"
                ),
                context_size=WandbParameter(256),
                pre_tokenized=WandbParameter(value=True),
                pre_download=WandbParameter(value=True),
                # Total dataset is c.7bn activations (64 files)
                dataset_files=WandbParameter(
                    [f"data/train-{str(i).zfill(5)}-of-00064.parquet" for i in range(22)]
                ),
            ),
            autoencoder=AutoencoderHyperparameters(
                expansion_factor=WandbParameter(values=[32, 64])
            ),
            pipeline=PipelineHyperparameters(),
            activation_resampler=ActivationResamplerHyperparameters(
                threshold_is_dead_portion_fires=WandbParameter(1e-5),
            ),
        ),
        method=Method.GRID,
    )

    sweep(sweep_config=sweep_config, project_name="gpt_2_small_sae_all_layers")


if __name__ == "__main__":
    train()
