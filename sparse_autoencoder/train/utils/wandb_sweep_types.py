"""Wandb Sweep Config Dataclasses.

Weights & Biases just provide a JSON Schema, so we've converted here to dataclasses.
"""
from abc import ABC
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum, auto
from typing import Any, Generic, TypeVar, final

from strenum import LowercaseStrEnum


class ControllerType(LowercaseStrEnum):
    """Controller Type."""

    CLOUD = auto()
    """Weights & Biases cloud controller.

    Utilizes Weights & Biases as the sweep controller, enabling launching of multiple nodes that all
    communicate with the Weights & Biases cloud service to coordinate the sweep.
    """

    LOCAL = auto()
    """Local controller.

    Manages the sweep operation locally, without the need for cloud-based coordination or external
    services.
    """


class HyperbandStoppingType(LowercaseStrEnum):
    """Hyperband Stopping Type."""

    HYPERBAND = auto()
    """Hyperband algorithm.

    Implements the Hyperband stopping algorithm, an adaptive resource allocation and early-stopping
    method to efficiently tune hyperparameters.
    """


class Kind(LowercaseStrEnum):
    """Kind."""

    SWEEP = auto()


class Method(LowercaseStrEnum):
    """Method."""

    BAYES = auto()
    """Bayesian optimization.

    Employs Bayesian optimization for hyperparameter tuning, a probabilistic model-based approach
    for finding the optimal set of parameters.
    """

    CUSTOM = auto()
    """Custom method.

    Allows for a user-defined custom method for hyperparameter tuning, providing flexibility in the
    sweep process.
    """

    GRID = auto()
    """Grid search.

    Utilizes a grid search approach for hyperparameter tuning, systematically working through
    multiple combinations of parameter values.
    """

    RANDOM = auto()
    """Random search.

    Implements a random search strategy for hyperparameter tuning, exploring the parameter space
    randomly.
    """


class Goal(LowercaseStrEnum):
    """Goal."""

    MAXIMIZE = auto()
    """Maximization goal.

    Sets the objective of the hyperparameter tuning process to maximize a specified metric.
    """

    MINIMIZE = auto()
    """Minimization goal.

    Aims to minimize a specified metric during the hyperparameter tuning process.
    """


class Impute(LowercaseStrEnum):
    """Metric value to use in bayes search for runs that fail, crash, or are killed."""

    BEST = auto()
    LATEST = auto()
    WORST = auto()


class ImputeWhileRunning(LowercaseStrEnum):
    """Appends a calculated metric even when epochs are in a running state."""

    BEST = auto()
    FALSE = auto()
    LATEST = auto()
    WORST = auto()


class Distribution(LowercaseStrEnum):
    """Sweep Distribution."""

    BETA = auto()
    """Beta distribution.

    Utilizes the Beta distribution, a family of continuous probability distributions defined on the
    interval [0, 1], for parameter sampling.
    """

    CATEGORICAL = auto()
    """Categorical distribution.

    Employs a categorical distribution for discrete variable sampling, where each category has an
    equal probability of being selected.
    """

    CATEGORICAL_W_PROBABILITIES = auto()
    """Categorical distribution with probabilities.

    Similar to categorical distribution but allows assigning different probabilities to each
    category.
    """

    CONSTANT = auto()
    """Constant distribution.

    Uses a constant value for the parameter, ensuring it remains the same across all runs.
    """

    INT_UNIFORM = auto()
    """Integer uniform distribution.

    Samples integer values uniformly across a specified range.
    """

    INV_LOG_UNIFORM = auto()
    """Inverse log-uniform distribution.

    Samples values according to an inverse log-uniform distribution, useful for parameters that span
    several orders of magnitude.
    """

    INV_LOG_UNIFORM_VALUES = auto()
    """Inverse log-uniform values distribution.

    Similar to the inverse log-uniform distribution but allows specifying exact values to be
    sampled.
    """


@dataclass
class Controller:
    """Controller."""

    type: ControllerType  # noqa: A003


@dataclass
class HyperbandStopping:
    """Hyperband Stopping Config.

    Speed up hyperparameter search by killing off runs that appear to have lower performance
    than successful training runs.

    Example:
        >>> HyperbandStopping(type=HyperbandStoppingType.HYPERBAND)
        HyperbandStopping(type=hyperband)
    """

    type: HyperbandStoppingType | None = HyperbandStoppingType.HYPERBAND  # noqa: A003

    eta: float | None = None
    """ETA.

    Specify the bracket multiplier schedule (default: 3).
    """

    maxiter: int | None = None
    """Max Iterations.

    Specify the maximum number of iterations. Note this is number of times the metric is logged, not
    the number of activations.
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

    @final
    def __str__(self) -> str:
        """String representation of this object."""
        items_representation = []
        for key, value in self.__dict__.items():
            if value is not None:
                items_representation.append(f"{key}={value}")
        joined_items = ", ".join(items_representation)

        class_name = self.__class__.__name__

        return f"{class_name}({joined_items})"

    @final
    def __repr__(self) -> str:
        """Representation of this object."""
        return self.__str__()


@dataclass(frozen=True)
class Metric:
    """Metric to optimize."""

    name: str
    """Name of metric."""

    goal: Goal | None = Goal.MINIMIZE

    impute: Impute | None = None
    """Metric value to use in bayes search for runs that fail, crash, or are killed"""

    imputewhilerunning: ImputeWhileRunning | None = None
    """Appends a calculated metric even when epochs are in a running state."""

    target: float | None = None
    """The sweep will finish once any run achieves this value."""

    @final
    def __str__(self) -> str:
        """String representation of this object."""
        items_representation = []
        for key, value in self.__dict__.items():
            if value is not None:
                items_representation.append(f"{key}={value}")
        joined_items = ", ".join(items_representation)

        class_name = self.__class__.__name__

        return f"{class_name}({joined_items})"

    @final
    def __repr__(self) -> str:
        """Representation of this object."""
        return self.__str__()


ParamType = TypeVar(
    "ParamType",
    float,
    int,
    str,
)


@dataclass(frozen=True)
class Parameter(Generic[ParamType]):
    """Sweep Parameter.

    https://docs.wandb.ai/guides/sweeps/define-sweep-configuration#parameters
    """

    value: ParamType | None = None
    """Single value.

    Specifies the single valid value for this hyperparameter. Compatible with grid.
    """

    max: ParamType | None = None  # noqa: A003
    """Maximum value."""

    min: ParamType | None = None  # noqa: A003
    """Minimum value."""

    distribution: Distribution | None = None
    """Distribution

    If not specified, will default to categorical if values is set, to int_uniform if max and min
    are set to integers, to uniform if max and min are set to floats, or to constant if value is
    set.
    """

    q: float | None = None
    """Quantization parameter.

    Quantization step size for quantized hyperparameters.
    """

    values: list[ParamType] | None = None
    """Discrete values.

    Specifies all valid values for this hyperparameter. Compatible with grid.
    """

    probabilities: list[float] | None = None
    """Probability of each value"""

    mu: float | None = None
    """Mean for normal or lognormal distributions"""

    sigma: float | None = None
    """Std Dev for normal or lognormal distributions"""

    @final
    def __str__(self) -> str:
        """String representation of this object."""
        items_representation = []
        for key, value in self.__dict__.items():
            if value is not None:
                items_representation.append(f"{key}={value}")
        joined_items = ", ".join(items_representation)

        class_name = self.__class__.__name__

        return f"{class_name}({joined_items})"

    @final
    def __repr__(self) -> str:
        """Representation of this object."""
        return self.__str__()


@dataclass(frozen=True)
class NestedParameter(ABC):  # noqa: B024 (abstract so that we can check against it's type)
    """Nested Parameter.

    Example:
        >>> from dataclasses import field
        >>> @dataclass(frozen=True)
        ... class MyNestedParameter(NestedParameter):
        ...     a: int = field(default=Parameter(1))
        ...     b: int = field(default=Parameter(2))
        >>> MyNestedParameter().to_dict()
        {'parameters': {'a': {'value': 1}, 'b': {'value': 2}}}
    """

    def to_dict(self) -> dict[str, Any]:
        """Return dict representation of this object."""

        def dict_without_none_values(obj: Any) -> dict:  # noqa: ANN401
            """Return dict without None values.

            Args:
                obj: The object to convert to a dict.

            Returns:
                The dict representation of the object.
            """
            dict_none_removed = {}
            dict_with_none = dict(obj)
            for key, value in dict_with_none.items():
                if value is not None:
                    dict_none_removed[key] = value
            return dict_none_removed

        return {"parameters": asdict(self, dict_factory=dict_without_none_values)}

    def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
        """Return dict representation of this object."""
        return self.to_dict()


@dataclass
class Parameters:
    """Parameters"""


@dataclass
class WandbSweepConfig:
    """Weights & Biases Sweep Configuration.

    Example:
        >>> config = WandbSweepConfig(
        ...     parameters={"lr": Parameter(value=1e-3)},
        ...     method=Method.BAYES,
        ...     metric=Metric(name="loss"),
        ...     )
        >>> print(config.to_dict()["parameters"])
        {'lr': {'value': 0.001}}
    """

    parameters: Parameters | Any

    method: Method
    """Method (search strategy)."""

    metric: Metric
    """Metric to optimize"""

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

    def to_dict(self) -> dict[str, Any]:
        """Return dict representation of this object.

        Recursively removes all None values. Handles special cases of dataclass
        instances and values that are `NestedParameter` instances.

        Returns:
            dict[str, Any]: The dict representation of the object.
        """

        def recursive_format(obj: Any) -> Any:  # noqa: ANN401
            """Recursively format the dict of hyperparameters."""
            # Handle dataclasses
            if is_dataclass(obj):
                cleaned_obj = {}
                for parameter_name in asdict(obj):
                    value = getattr(obj, parameter_name)

                    # Remove None values.
                    if value is None:
                        continue

                    # Nested parameters have their own `to_dict` method, which we can call.
                    if isinstance(value, NestedParameter):
                        cleaned_obj[parameter_name] = value.to_dict()
                    # Otherwise recurse.
                    else:
                        cleaned_obj[parameter_name] = recursive_format(value)
                return cleaned_obj

            # Handle dicts
            if isinstance(obj, dict):
                cleaned_obj = {}
                for key, value in obj.items():
                    # Remove None values.
                    if value is None:
                        continue

                    # Otherwise recurse.
                    cleaned_obj[key] = recursive_format(value)
                return cleaned_obj

            # Handle enums
            if isinstance(obj, Enum):
                return obj.value

            # Handle other types (e.g. float, int, str)
            return obj

        return recursive_format(self)

    def __dict__(self) -> dict[str, Any]:  # type: ignore[override]
        """Return dict representation of this object."""
        return self.to_dict()
