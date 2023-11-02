"""Sweep Config."""
from dataclasses import dataclass, field

from sparse_autoencoder.train.utils.results_dataclass import (
    convert_parameters_to_results_type,
)
from sparse_autoencoder.train.utils.wandb_sweep_types import (
    Method,
    Parameter,
    Parameters,
    WandbSweepConfig,
)


@dataclass
class SweepParameterConfig(Parameters):
    """Sweep Parameter Config."""

    lr: Parameter[float] = field(default_factory=lambda: Parameter(value=0.001))
    """Adam Learning Rate."""

    adam_beta_1: Parameter[float] = field(default_factory=lambda: Parameter(value=0.9))
    """Adam Beta 1.
    
    The exponential decay rate for the first moment estimates (mean) of the gradient.
    """

    adam_beta_2: Parameter[float] = field(
        default_factory=lambda: Parameter(value=0.999)
    )
    """Adam Beta 2.
    
    The exponential decay rate for the second moment estimates (variance) of the gradient.
    """

    adam_epsilon: Parameter[float] = field(
        default_factory=lambda: Parameter(value=1e-8)
    )
    """Adam Epsilon.
    
    A small constant for numerical stability.
    """

    adam_weight_decay: Parameter[float] = field(
        default_factory=lambda: Parameter(value=0)
    )
    """Adam Weight Decay.
    
    Weight decay (L2 penalty).
    """

    l1_coefficient: Parameter[float] = field(
        default_factory=lambda: Parameter(value=[0.001, 0.004, 0.006, 0.008, 1])
    )
    """L1 Penalty Coefficient.
    
    The L1 penalty is the absolute sum of learned (hidden) activations, multiplied by this constant.
    The penalty encourages sparsity in the learned activations. This loss penalty can be reduced by
    using more features, or using a lower L1 coefficient.
    
    Default values from the [original
    paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
    """

    width_multiplier: Parameter[int] = field(
        default_factory=lambda: Parameter(value=8, min=1, max=256)
    )
    """Source-to-Trained Activations Width Multiplier."""


SweepRunParameters = convert_parameters_to_results_type(SweepParameterConfig)


@dataclass
class SweepConfig(WandbSweepConfig):
    """Sweep Config."""

    parameters: SweepParameterConfig

    method: Method = Method.bayes
