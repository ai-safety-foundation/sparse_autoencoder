"""Wandb Sweep Config Dataclasses.

Weights & Biases just provide a JSON Schema, so we've converted here to dataclasses.
"""
# ruff: noqa
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, TypeVar


class ControllerType(Enum):
    """Controller Type."""

    cloud = "cloud"
    local = "local"


@dataclass(frozen=True)
class Controller:
    """Controller."""

    type: ControllerType  # noqa: A003


class HyperbandStoppingType(Enum):
    """Hyperband Stopping Type."""

    hyperband = "hyperband"


@dataclass(frozen=True)
class HyperbandStopping:
    """Hyperband Stopping Config.

    Speed up hyperparameter search by killing off runs that appear to have lower performance
    than successful training runs.
    """

    type: HyperbandStoppingType  # noqa: A003

    eta: float | None = None
    """ETA.

    At every eta^n steps, hyperband continues running the top 1/eta runs and stops all other
    runs.
    """

    maxiter: int | None = None
    """Max Iterations.

    Set the last epoch to finish trimming runs, and hyperband will automatically calculate
    the prior epochs to trim runs.
    """

    miniter: int | None = None
    """Min Iterations.

    Set the first epoch to start trimming runs, and hyperband will automatically calculate
    the subsequent epochs to trim runs.
    """

    s: float | None = None
    """Set the number of steps you trim runs at, working backwards from the max_iter."""

    strict: bool | None = None
    """Use a more aggressive condition for termination, stops more runs."""


class Kind(Enum):
    """Kind."""

    sweep = "sweep"


class Method(Enum):
    """Method."""

    bayes = "bayes"
    custom = "custom"
    grid = "grid"
    random = "random"


class Goal(Enum):
    """Goal."""

    maximize = "maximize"
    minimize = "minimize"


class Impute(Enum):
    """Metric value to use in bayes search for runs that fail, crash, or are killed."""

    best = "best"
    latest = "latest"
    worst = "worst"


class ImputeWhileRunning(Enum):
    """Appends a calculated metric even when epochs are in a running state."""

    best = "best"
    false = "false"
    latest = "latest"
    worst = "worst"


@dataclass(frozen=True)
class Metric:
    """Metric to optimize."""

    name: str
    """Name of metric."""

    goal: Goal | None = None

    impute: Impute | None = None
    """Metric value to use in bayes search for runs that fail, crash, or are killed"""

    imputewhilerunning: ImputeWhileRunning | None = None
    """Appends a calculated metric even when epochs are in a running state."""

    target: float | None = None
    """The sweep will finish once any run achieves this value."""


class Distribution(Enum):
    """Sweep Distribution."""

    beta = "beta"
    categorical = "categorical"
    categoricalwprobabilities = "categorical_w_probabilities"
    constant = "constant"
    intuniform = "int_uniform"
    invloguniform = "inv_log_uniform"
    invloguniformvalues = "inv_log_uniform_values"
    lognormal = "log_normal"
    loguniform = "log_uniform"
    loguniformvalues = "log_uniform_values"
    normal = "normal"
    qbeta = "q_beta"
    qlognormal = "q_log_normal"
    qloguniform = "q_log_uniform"
    qloguniformvalues = "q_log_uniform_values"
    qnormal = "q_normal"
    quniform = "q_uniform"
    uniform = "uniform"


ParamType = TypeVar("ParamType", float, int, str)


@dataclass(frozen=True)
class Parameter(Generic[ParamType]):
    """Sweep Parameter."""

    value: ParamType | list[ParamType]

    max: ParamType | None = None  # noqa: A003

    min: ParamType | None = None  # noqa: A003

    a: float | None = None

    b: float | None = None

    distribution: Distribution | None = None

    q: float | None = None
    """Quantization parameter for quantized distributions"""

    values: list[ParamType] | None = None
    """Discrete values"""

    probabilities: list[float] | None = None
    """Probability of each value"""

    mu: float | None = None
    """Mean for normal or lognormal distributions"""

    sigma: float | None = None
    """Std Dev for normal or lognormal distributions"""

    parameters: dict[str, "Parameter[ParamType]"] | None = None


Parameters = dict[str, Parameter[Any]]


@dataclass(frozen=True)
class WandbSweepConfig:
    """Weights & Biases Sweep Configuration."""

    parameters: Parameters | Any

    method: Method

    metric: Metric
    """Metric to optimize"""

    apiVersion: str | None = None

    command: list[Any] | None = None
    """Command used to launch the training script"""

    controller: Controller | None = None

    description: str | None = None
    """Short package description"""

    earlyterminate: HyperbandStopping | None = None

    entity: str | None = None
    """The entity for this sweep"""

    imageuri: str | None = None
    """Sweeps on Launch will use this uri instead of a job."""

    job: str | None = None
    """Launch Job to run."""

    kind: Kind | None = None

    name: str | None = None
    """The name of the sweep, displayed in the W&B UI."""

    program: str | None = None
    """Training script to run."""

    project: str | None = None
    """The project for this sweep."""

    runcap: int | None = None
    """Run Cap.

    Sweep will run no more than this number of runs, across any number of agents.
    """
