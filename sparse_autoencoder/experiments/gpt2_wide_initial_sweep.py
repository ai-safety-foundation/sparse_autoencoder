"""Run an initial sweep on GPT 2 Small's first layer."""
import os

from sparse_autoencoder import (
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
from sparse_autoencoder.train.sweep_config import ActivationResamplerHyperparameters


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main() -> None:
    """Run the experiment."""
    sweep_config = SweepConfig(
        parameters=Hyperparameters(
            loss=LossHyperparameters(
                l1_coefficient=Parameter(min=5e-4, max=1e-2),
            ),
            optimizer=OptimizerHyperparameters(
                lr=Parameter(min=1e-4, max=1e-3),
            ),
            source_model=SourceModelHyperparameters(
                name=Parameter("gpt2"),
                cache_names=Parameter(["blocks.0.hook_mlp_out"]),
                hook_dimension=Parameter(768),
            ),
            source_data=SourceDataHyperparameters(
                dataset_path=Parameter("alancooney/sae-monology-pile-uncopyrighted-tokenizer-gpt2"),
                context_size=Parameter(128),
                pre_tokenized=Parameter(value=True),
            ),
            autoencoder=AutoencoderHyperparameters(expansion_factor=Parameter(values=[2, 4, 8])),
            pipeline=PipelineHyperparameters(
                max_activations=Parameter(1_000_000_000),
                checkpoint_frequency=Parameter(100_000_000),
                validation_frequency=Parameter(100_000_000),
            ),
            activation_resampler=ActivationResamplerHyperparameters(
                resample_interval=Parameter(200_000_000),
                n_activations_activity_collate=Parameter(100_000_000),
                threshold_is_dead_portion_fires=Parameter(
                    1e-6,
                ),
                max_n_resamples=Parameter(4),
            ),
        ),
        method=Method.RANDOM,
    )

    sweep(sweep_config=sweep_config)


if __name__ == "__main__":
    main()
