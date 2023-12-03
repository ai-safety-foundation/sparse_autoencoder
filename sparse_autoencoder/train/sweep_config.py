"""Default hyperparameter setup for quick tuning of a sparse autoencoder."""
from dataclasses import dataclass, field
from typing import TypedDict

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


class OptimizerRuntimeHyperparameters(TypedDict):
    """Optimizer runtime hyperparameters."""

    lr: float
    adam_beta_1: float
    adam_beta_2: float
    adam_weight_decay: float


@dataclass
class SourceDataHyperparameters(NestedParameter):
    """Source data hyperparameters."""

    context_size: Parameter[int] = field(default_factory=lambda: Parameter(128))
    """Context size."""


class SourceDataRuntimeHyperparameters(TypedDict):
    """Source data runtime hyperparameters."""

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


class PipelineRuntimeHyperparameters(TypedDict):
    """Pipeline runtime hyperparameters."""

    train_batch_size: int
    max_store_size: int
    max_activations: int
    checkpoint_frequency: int
    validation_frequency: int


@dataclass
class Hyperparameters(Parameters):
    """Sweep Hyperparameters."""

    source_model: SourceModelHyperparameters

    autoencoder: AutoencoderHyperparameters = field(
        default_factory=lambda: AutoencoderHyperparameters()
    )

    loss: LossHyperparameters = field(default_factory=lambda: LossHyperparameters())

    optimizer: OptimizerHyperparameters = field(default_factory=lambda: OptimizerHyperparameters())

    source_data: SourceDataHyperparameters = field(
        default_factory=lambda: SourceDataHyperparameters()
    )

    pipeline: PipelineHyperparameters = field(default_factory=lambda: PipelineHyperparameters())

    random_seed: Parameter[int] = field(default_factory=lambda: Parameter(49))
    """Random seed."""


@dataclass
class SweepConfig(WandbSweepConfig):
    """Sweep Config."""

    parameters: Hyperparameters

    method: Method = Method.RANDOM

    metric: Metric = field(default_factory=lambda: Metric(name="total_loss"))


class RuntimeHyperparameters(TypedDict):
    """Runtime hyperparameters."""

    source_model: SourceModelRuntimeHyperparameters
    autoencoder: AutoencoderRuntimeHyperparameters
    loss: LossRuntimeHyperparameters
    optimizer: OptimizerRuntimeHyperparameters
    source_data: SourceDataRuntimeHyperparameters
    pipeline: PipelineRuntimeHyperparameters
    random_seed: int
