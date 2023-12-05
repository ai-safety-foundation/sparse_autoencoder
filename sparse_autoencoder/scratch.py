"""Scratch."""
from sparse_autoencoder import (
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
from sparse_autoencoder.train.sweep_config import (
    DEFAULT_STORE_SIZE,
    ActivationResamplerHyperparameters,
    AutoencoderHyperparameters,
)
from sparse_autoencoder.train.utils.round_down import round_to_multiple


sweep_config = SweepConfig(
    parameters=Hyperparameters(
        loss=LossHyperparameters(
            l1_coefficient=Parameter(max=1e-2, min=4e-3),
        ),
        optimizer=OptimizerHyperparameters(
            lr=Parameter(max=1e-3, min=1e-5),
        ),
        source_model=SourceModelHyperparameters(
            name=Parameter("gelu-2l"),
            hook_site=Parameter("mlp_out"),
            hook_layer=Parameter(0),
            hook_dimension=Parameter(512),
        ),
        source_data=SourceDataHyperparameters(
            dataset_path=Parameter("NeelNanda/c4-code-tokenized-2b"),
        ),
        pipeline=PipelineHyperparameters(
            max_activations=Parameter(round_to_multiple(200_000_000, DEFAULT_STORE_SIZE)),
            validation_frequency=Parameter(round_to_multiple(25_000_000, DEFAULT_STORE_SIZE)),
        ),
        autoencoder=AutoencoderHyperparameters(
            expansion_factor=Parameter(4),
        ),
        activation_resampler=ActivationResamplerHyperparameters(
            resample_interval=Parameter(round_to_multiple(10_000_000, DEFAULT_STORE_SIZE))
        ),
    ),
    method=Method.BAYES,
    runcap=10,
)

if __name__ == "__main__":
    sweep(sweep_config=sweep_config)

    # l1 0.008, lr 8e4 = good
    # l1 0.009, lr 1e4 = good
    # l1 0.003, lr 3e4 = bad
    # All low reconstruction loss
