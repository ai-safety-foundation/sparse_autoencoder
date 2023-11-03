"""Sweep Config."""
from dataclasses import asdict, dataclass, field

from sparse_autoencoder.train.utils.wandb_sweep_types import (
    Method,
    Metric,
    Parameter,
    Parameters,
    WandbSweepConfig,
)


# NOTE: This must be kept in sync with SweepParametersRuntime
@dataclass
class SweepParameterConfig(Parameters):
    """Sweep Parameter Config."""

    lr: Parameter[float] | None
    """Adam Learning Rate."""

    adam_beta_1: Parameter[float] | None
    """Adam Beta 1.
    
    The exponential decay rate for the first moment estimates (mean) of the gradient.
    """

    adam_beta_2: Parameter[float] | None
    """Adam Beta 2.
    
    The exponential decay rate for the second moment estimates (variance) of the gradient.
    """

    adam_epsilon: Parameter[float] | None
    """Adam Epsilon.
    
    A small constant for numerical stability.
    """

    adam_weight_decay: Parameter[float] | None
    """Adam Weight Decay.
    
    Weight decay (L2 penalty).
    """

    batch_size: Parameter[int] | None

    l1_coefficient: Parameter[float] | None
    """L1 Penalty Coefficient.
    
    The L1 penalty is the absolute sum of learned (hidden) activations, multiplied by this constant.
    The penalty encourages sparsity in the learned activations. This loss penalty can be reduced by
    using more features, or using a lower L1 coefficient.
    
    Default values from the [original
    paper](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
    """


# NOTE: This must be kept in sync with SweepParameterConfig
@dataclass
class SweepParametersRuntime(dict):
    """Sweep parameter runtime values."""

    lr: float = 0.001

    adam_beta_1: float = 0.9

    adam_beta_2: float = 0.999

    adam_epsilon: float = 1e-8

    adam_weight_decay: float = 0.0

    batch_size: int = 2

    l1_coefficient: float = 0.01

    def to_dict(self):
        """Return dict representation of this object."""
        return asdict(self)


@dataclass
class SweepConfig(WandbSweepConfig):
    """Sweep Config."""

    parameters: SweepParameterConfig

    method: Method = Method.grid

    metric: Metric = field(default_factory=lambda: Metric(name="loss"))

    def to_dict(self):
        """Return dict representation of this object."""
        dict_representation = asdict(self)

        # Convert StrEnums to strings
        dict_representation["method"] = dict_representation["method"].value

        def remove_none_values(d):
            """Recursively remove all None values from the dictionary."""
            if isinstance(d, dict):
                return {
                    k: remove_none_values(v)
                    for k, v in d.items()
                    if v is not None and remove_none_values(v) is not None
                }
            elif isinstance(d, list):
                return [
                    remove_none_values(v)
                    for v in d
                    if v is not None and remove_none_values(v) is not None
                ]
            else:
                return d

        return remove_none_values(dict_representation)
