"""Sweep config.

Default hyperparameter setup for quick tuning of a sparse autoencoder.

Warning:
    The runtime hyperparameter classes must be manually kept in sync with the hyperparameter
    classes, so that static type checking works.
"""
from dataclasses import dataclass, field
from typing import TypedDict, final

from sparse_autoencoder.train.utils.round_down import round_to_multiple
from sparse_autoencoder.train.utils.wandb_sweep_types import (
    Method,
    Metric,
    NestedParameter,
    Parameter,
    Parameters,
    WandbSweepConfig,
)


# Key default values (used to calculate other default values)
DEFAULT_SOURCE_BATCH_SIZE: int = 16
DEFAULT_SOURCE_CONTEXT_SIZE: int = 256
DEFAULT_BATCH_SIZE: int = 8192  # Should be a multiple of source batch size and context size
DEFAULT_STORE_SIZE: int = round_to_multiple(3_000_000, DEFAULT_BATCH_SIZE)


@dataclass(frozen=True)
class ActivationResamplerHyperparameters(NestedParameter):
    """Activation resampler hyperparameters."""

    resample_interval: Parameter[int] = field(
        default=Parameter(round_to_multiple(200_000_000, DEFAULT_STORE_SIZE))
    )
    """Resample interval."""

    max_n_resamples: Parameter[int] = field(default=Parameter(4))
    """Maximum number of resamples."""

    n_activations_activity_collate: Parameter[int] = field(
        default=Parameter(round_to_multiple(100_000_000, DEFAULT_STORE_SIZE))
    )
    """Number of steps to collate before resampling.

    Number of autoencoder learned activation vectors to collate before resampling.
    """

    resample_dataset_size: Parameter[int] = field(default=Parameter(DEFAULT_BATCH_SIZE * 100))
    """Resample dataset size.

    Number of autoencoder input activations to use for calculating the loss, as part of the
    resampling process to create the reset neuron weights.
    """

    threshold_is_dead_portion_fires: Parameter[float] = field(default=Parameter(0.0))
    """Dead neuron threshold.

    Threshold for determining if a neuron is dead (has "fired" in less than this portion of the
    collated sample).
    """


class ActivationResamplerRuntimeHyperparameters(TypedDict):
    """Activation resampler runtime hyperparameters."""

    resample_interval: int
    max_n_resamples: int
    n_activations_activity_collate: int
    resample_dataset_size: int
    threshold_is_dead_portion_fires: float


@dataclass(frozen=True)
class AutoencoderHyperparameters(NestedParameter):
    """Sparse autoencoder hyperparameters."""

    expansion_factor: Parameter[int] = field(default=Parameter(2))
    """Expansion Factor.

    Size of the learned features relative to the input features. A good expansion factor to start
    with is typically 2-4.
    """


class AutoencoderRuntimeHyperparameters(TypedDict):
    """Autoencoder runtime hyperparameters."""

    expansion_factor: int


@dataclass(frozen=True)
class LossHyperparameters(NestedParameter):
    """Loss hyperparameters."""

    l1_coefficient: Parameter[float] = field(default=Parameter(1e-3))
    """L1 Penalty Coefficient.

    The L1 penalty is the absolute sum of learned (hidden) activations, multiplied by this constant.
    The penalty encourages sparsity in the learned activations. This loss penalty can be reduced by
    using more features, or using a lower L1 coefficient. If your expansion factor is 2, then a good
    starting point for the L1 coefficient is 1e-3.
    """


class LossRuntimeHyperparameters(TypedDict):
    """Loss runtime hyperparameters."""

    l1_coefficient: float


@dataclass(frozen=True)
class OptimizerHyperparameters(NestedParameter):
    """Optimizer hyperparameters."""

    lr: Parameter[float] = field(default=Parameter(1e-3))
    """Learning rate.

    A good starting point for the learning rate is 1e-3, but this is one of the key parameters so
    you should probably tune it.
    """

    adam_beta_1: Parameter[float] = field(default=Parameter(0.9))
    """Adam Beta 1.

    The exponential decay rate for the first moment estimates (mean) of the gradient.
    """

    adam_beta_2: Parameter[float] = field(default=Parameter(0.99))
    """Adam Beta 2.

    The exponential decay rate for the second moment estimates (variance) of the gradient.
    """

    adam_weight_decay: Parameter[float] = field(default=Parameter(0.0))
    """Adam Weight Decay.

    Weight decay (L2 penalty).
    """

    amsgrad: Parameter[bool] = field(default=Parameter(value=False))
    """AMSGrad.

    Whether to use the AMSGrad variant of this algorithm from the paper [On the Convergence of Adam
    and Beyond](https://arxiv.org/abs/1904.09237).
    """

    fused: Parameter[bool] = field(default=Parameter(value=False))
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


@dataclass(frozen=True)
class SourceDataHyperparameters(NestedParameter):
    """Source data hyperparameters."""

    dataset_path: Parameter[str]
    """Dataset path."""

    context_size: Parameter[int] = field(default=Parameter(DEFAULT_SOURCE_CONTEXT_SIZE))
    """Context size."""

    dataset_dir: Parameter[str] | None = field(default=None)
    """Dataset directory (within the HF dataset)"""

    dataset_files: Parameter[str] | None = field(default=None)
    """Dataset files (within the HF dataset)."""

    pre_tokenized: Parameter[bool] = field(default=Parameter(value=True))
    """If the dataset is pre-tokenized."""

    tokenizer_name: Parameter[str] | None = field(default=None)
    """Tokenizer name.

    Only set this if the dataset is not pre-tokenized.
    """

    def __post_init__(self) -> None:
        """Post initialisation checks.

        Raises:
            ValueError: If there is an error in the source data hyperparameters.
        """
        if self.pre_tokenized.value is False and not isinstance(self.tokenizer_name, Parameter):
            error_message = "The tokenizer name must be specified, when `pre_tokenized` is False."
            raise ValueError(error_message)

        if self.pre_tokenized.value is True and isinstance(self.tokenizer_name, Parameter):
            error_message = "The tokenizer name must not be set, when `pre_tokenized` is True."
            raise ValueError(error_message)


class SourceDataRuntimeHyperparameters(TypedDict):
    """Source data runtime hyperparameters."""

    context_size: int
    dataset_dir: str | None
    dataset_files: str | None
    dataset_path: str
    pre_tokenized: bool
    tokenizer_name: str | None


@dataclass(frozen=True)
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

    dtype: Parameter[str] = field(default=Parameter("float32"))
    """Source model dtype."""


class SourceModelRuntimeHyperparameters(TypedDict):
    """Source model runtime hyperparameters."""

    name: str
    hook_site: str
    hook_layer: int
    hook_dimension: int
    dtype: str


@dataclass(frozen=True)
class PipelineHyperparameters(NestedParameter):
    """Pipeline hyperparameters."""

    log_frequency: Parameter[int] = field(default=Parameter(100))
    """Training log frequency."""

    source_data_batch_size: Parameter[int] = field(default=Parameter(DEFAULT_SOURCE_BATCH_SIZE))
    """Source data batch size."""

    train_batch_size: Parameter[int] = field(default=Parameter(DEFAULT_BATCH_SIZE))
    """Train batch size."""

    max_store_size: Parameter[int] = field(default=Parameter(DEFAULT_STORE_SIZE))
    """Max store size."""

    max_activations: Parameter[int] = field(
        default=Parameter(round_to_multiple(2e9, DEFAULT_STORE_SIZE))
    )
    """Max activations."""

    checkpoint_frequency: Parameter[int] = field(
        default=Parameter(round_to_multiple(5e7, DEFAULT_STORE_SIZE))
    )
    """Checkpoint frequency."""

    validation_frequency: Parameter[int] = field(
        default=Parameter(round_to_multiple(1e8, DEFAULT_BATCH_SIZE))
    )
    """Validation frequency."""

    validation_number_activations: Parameter[int] = field(
        # Default to a single batch of source data prompts
        default=Parameter(DEFAULT_BATCH_SIZE * DEFAULT_SOURCE_CONTEXT_SIZE * 16)
    )
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
        default=ActivationResamplerHyperparameters()
    )

    autoencoder: AutoencoderHyperparameters = field(default=AutoencoderHyperparameters())

    loss: LossHyperparameters = field(default=LossHyperparameters())

    optimizer: OptimizerHyperparameters = field(default=OptimizerHyperparameters())

    pipeline: PipelineHyperparameters = field(default=PipelineHyperparameters())

    random_seed: Parameter[int] = field(default=Parameter(49))
    """Random seed."""

    def __post_init__(self) -> None:
        """Post initialisation checks."""
        # Check the resample dataset size <= the store size (currently only works if value is used
        # for both).
        if (
            self.activation_resampler.resample_dataset_size.value is not None
            and self.pipeline.max_store_size.value is not None
            and self.activation_resampler.resample_dataset_size.value
            > int(self.pipeline.max_store_size.value)
        ):
            error_message = (
                "Resample dataset size must be less than or equal to the pipeline max store size. "
                f"Resample dataset size: {self.activation_resampler.resample_dataset_size.value}, "
                f"pipeline max store size: {self.pipeline.max_store_size.value}."
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

    method: Method = Method.GRID

    metric: Metric = field(default=Metric(name="train/loss/total_loss"))


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
