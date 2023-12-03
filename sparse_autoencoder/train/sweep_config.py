"""Sweep config.

Default hyperparameter setup for quick tuning of a sparse autoencoder.
"""
from dataclasses import dataclass, field
from typing import TypedDict, final

from sparse_autoencoder.train.utils.wandb_sweep_types import (
    Method,
    Metric,
    NestedParameter,
    Parameter,
    Parameters,
    WandbSweepConfig,
)


# Warning: The runtime hyperparameter classes must be manually kept in sync with the hyperparameter
# classes, so that static type checking works.


@dataclass
class ActivationResamplerHyperparameters(NestedParameter):
    """Activation resampler hyperparameters."""

    resample_interval: Parameter[int] = field(default_factory=lambda: Parameter(200_000_000))
    """Resample interval."""

    max_resamples: Parameter[int] = field(default_factory=lambda: Parameter(4))
    """Maximum number of resamples."""

    n_steps_collate: Parameter[int] = field(default_factory=lambda: Parameter(100_000_000))
    """Number of steps to collate before resampling.

    Number of autoencoder learned activation vectors to collate before resampling.
    """

    resample_dataset_size: Parameter[int] = field(default_factory=lambda: Parameter(819_200))
    """Resample dataset size.

    Number of autoencoder input activations to use for calculating the loss, as part of the
    resampling process to create the reset neuron weights.
    """

    dead_neuron_threshold: Parameter[float] = field(default_factory=lambda: Parameter(0.0))
    """Dead neuron threshold.

    Threshold for determining if a neuron is dead (has "fired" in less than this portion of the
    collated sample).
    """


class ActivationResamplerRuntimeHyperparameters(TypedDict):
    """Activation resampler runtime hyperparameters."""

    resample_interval: int
    max_resamples: int
    n_steps_collate: int
    resample_dataset_size: int
    dead_neuron_threshold: float


@dataclass
class AutoencoderHyperparameters(NestedParameter):
    """Sparse autoencoder hyperparameters."""

    expansion_factor: Parameter[int] = field(default_factory=lambda: Parameter(4))
    """Expansion Factor.

    Size of the learned features relative to the input features.
    """


class AutoencoderRuntimeHyperparameters(TypedDict):
    """Autoencoder runtime hyperparameters."""

    expansion_factor: int


@dataclass
class LossHyperparameters(NestedParameter):
    """Loss hyperparameters."""

    l1_coefficient: Parameter[float] = field(default_factory=lambda: Parameter(1e-4))
    """L1 Penalty Coefficient.

    The L1 penalty is the absolute sum of learned (hidden) activations, multiplied by this constant.
    The penalty encourages sparsity in the learned activations. This loss penalty can be reduced by
    using more features, or using a lower L1 coefficient.
    """


class LossRuntimeHyperparameters(TypedDict):
    """Loss runtime hyperparameters."""

    l1_coefficient: float


@dataclass
class OptimizerHyperparameters(NestedParameter):
    """Optimizer hyperparameters."""

    lr: Parameter[float] = field(default_factory=lambda: Parameter(values=[1e-3, 1e-4, 1e-5]))
    """Learning rate."""

    adam_beta_1: Parameter[float] = field(default_factory=lambda: Parameter(0.9))
    """Adam Beta 1.

    The exponential decay rate for the first moment estimates (mean) of the gradient.
    """

    adam_beta_2: Parameter[float] = field(default_factory=lambda: Parameter(0.99))
    """Adam Beta 2.

    The exponential decay rate for the second moment estimates (variance) of the gradient.
    """

    adam_weight_decay: Parameter[float] = field(default_factory=lambda: Parameter(0.0))
    """Adam Weight Decay.

    Weight decay (L2 penalty).
    """

    amsgrad: Parameter[bool] = field(default_factory=lambda: Parameter(value=False))
    """AMSGrad.

    Whether to use the AMSGrad variant of this algorithm from the paper [On the Convergence of Adam
    and Beyond](https://arxiv.org/abs/1904.09237).
    """

    fused: Parameter[bool] = field(default_factory=lambda: Parameter(value=False))
    """Fused.

    Whether to use a fused implementation of the optimizer (may be faster on CUDA).
    """


class OptimizerRuntimeHyperparameters(TypedDict):
    """Optimizer runtime hyperparameters."""

    lr: float
    adam_beta_1: float
    adam_beta_2: float
    adam_weight_decay: float
    amsgrad: bool
    fused: bool


@dataclass
class SourceDataHyperparameters(NestedParameter):
    """Source data hyperparameters."""

    dataset_path: Parameter[str]
    """Dataset path."""

    context_size: Parameter[int] = field(default_factory=lambda: Parameter(128))
    """Context size."""


class SourceDataRuntimeHyperparameters(TypedDict):
    """Source data runtime hyperparameters."""

    dataset_path: str
    context_size: int


@dataclass
class SourceModelHyperparameters(NestedParameter):
    """Source model hyperparameters."""

    name: Parameter[str]
    """Source model name."""

    hook_site: Parameter[str]
    """Source model hook site."""

    hook_layer: Parameter[int]
    """Source model hook point layer."""

    hook_dimension: Parameter[int]
    """Source model hook point dimension."""

    dtype: Parameter[str] = field(default_factory=lambda: Parameter("float32"))
    """Source model dtype."""


class SourceModelRuntimeHyperparameters(TypedDict):
    """Source model runtime hyperparameters."""

    name: str
    hook_site: str
    hook_layer: int
    hook_dimension: int
    dtype: str


@dataclass
class PipelineHyperparameters(NestedParameter):
    """Pipeline hyperparameters."""

    log_frequency: Parameter[int] = field(default_factory=lambda: Parameter(100))
    """Training log frequency."""

    source_data_batch_size: Parameter[int] = field(default_factory=lambda: Parameter(12))
    """Source data batch size."""

    train_batch_size: Parameter[int] = field(default_factory=lambda: Parameter(4096))
    """Train batch size."""

    max_store_size: Parameter[int] = field(default_factory=lambda: Parameter(384 * 4096 * 2))
    """Max store size."""

    max_activations: Parameter[int] = field(default_factory=lambda: Parameter(2_000_000_000))
    """Max activations."""

    checkpoint_frequency: Parameter[int] = field(default_factory=lambda: Parameter(100_000_000))
    """Checkpoint frequency."""

    validation_frequency: Parameter[int] = field(
        default_factory=lambda: Parameter(384 * 4096 * 2 * 100)
    )
    """Validation frequency."""

    validation_number_activations: Parameter[int] = field(default_factory=lambda: Parameter(1024))
    """Number of activations to use for validation."""


class PipelineRuntimeHyperparameters(TypedDict):
    """Pipeline runtime hyperparameters."""

    log_frequency: int
    source_data_batch_size: int
    train_batch_size: int
    max_store_size: int
    max_activations: int
    checkpoint_frequency: int
    validation_frequency: int
    validation_number_activations: int


@dataclass
class Hyperparameters(Parameters):
    """Sweep Hyperparameters."""

    # Required parameters
    source_data: SourceDataHyperparameters

    source_model: SourceModelHyperparameters

    # Optional parameters
    activation_resampler: ActivationResamplerHyperparameters = field(
        default_factory=lambda: ActivationResamplerHyperparameters()
    )

    autoencoder: AutoencoderHyperparameters = field(
        default_factory=lambda: AutoencoderHyperparameters()
    )

    loss: LossHyperparameters = field(default_factory=lambda: LossHyperparameters())

    optimizer: OptimizerHyperparameters = field(default_factory=lambda: OptimizerHyperparameters())

    pipeline: PipelineHyperparameters = field(default_factory=lambda: PipelineHyperparameters())

    random_seed: Parameter[int] = field(default_factory=lambda: Parameter(49))
    """Random seed."""

    def __post_init__(self) -> None:
        """Post initialisation checks."""
        # Check the resample dataset size <= the store size (currently only works if value is used
        # for both).
        if (
            self.activation_resampler.resample_dataset_size.value is not None
            and self.pipeline.max_store_size.value is not None
            and self.activation_resampler.resample_dataset_size.value
            >= int(self.pipeline.max_store_size.value)
        ):
            error_message = (
                "Resample dataset size must be less than or equal to the pipeline max store size"
            )
            raise ValueError(error_message)

    @final
    def __str__(self) -> str:
        """String representation of this object."""
        items_representation = []
        for key, value in self.__dict__.items():
            if value is not None:
                items_representation.append(f"{key}={value}")
        joined_items = "\n    ".join(items_representation)

        class_name = self.__class__.__name__

        return f"{class_name}(\n    {joined_items}\n)"

    @final
    def __repr__(self) -> str:
        """Representation of this object."""
        return self.__str__()


@dataclass
class SweepConfig(WandbSweepConfig):
    """Sweep Config."""

    parameters: Hyperparameters

    method: Method = Method.RANDOM

    metric: Metric = field(default_factory=lambda: Metric(name="total_loss"))


class RuntimeHyperparameters(TypedDict):
    """Runtime hyperparameters."""

    source_data: SourceDataRuntimeHyperparameters
    source_model: SourceModelRuntimeHyperparameters
    activation_resampler: ActivationResamplerRuntimeHyperparameters
    autoencoder: AutoencoderRuntimeHyperparameters
    loss: LossRuntimeHyperparameters
    optimizer: OptimizerRuntimeHyperparameters
    pipeline: PipelineRuntimeHyperparameters
    random_seed: int
