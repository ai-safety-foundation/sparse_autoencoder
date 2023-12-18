"""Abstract metric."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto
from typing import Any, TypeAlias, cast, final

from jaxtyping import Float, Int
import numpy as np
from strenum import LowercaseStrEnum, SnakeCaseStrEnum
import torch
from torch import Tensor
from wandb import data_types

from sparse_autoencoder.tensor_types import Axis


class MetricLocation(SnakeCaseStrEnum):
    """Metric type name."""

    GENERATE = auto()
    TRAIN = auto()
    RESAMPLE = auto()
    VALIDATE = auto()
    SAVE = auto()


class ComponentAggregationApproach(LowercaseStrEnum):
    """Component aggregation method."""

    MEAN = auto()
    MAX = auto()
    MIN = auto()
    SUM = auto()
    TABLE = auto()


DEFAULT_EMPTY_LIST = []


WandbSupportedLogTypes: TypeAlias = (
    bool
    | data_types.Audio
    | data_types.Bokeh
    | data_types.Histogram
    | data_types.Html
    | data_types.Image
    | data_types.Molecule
    | data_types.Object3D
    | data_types.Plotly
    | data_types.Table
    | data_types.Video
    | data_types.WBTraceTree
    | float
    | Float[Tensor, Axis.names(Axis.SINGLE_ITEM)]
    | int
    | Int[Tensor, Axis.names(Axis.SINGLE_ITEM)]
    | list["WandbSupportedLogTypes"]
    | np.ndarray
)


@dataclass
class MetricInputData(ABC):  # noqa: B024
    """Metric input data."""

    @final
    @staticmethod
    def add_component_axis_if_missing(
        input_tensor: Float[Tensor, Axis.names(Axis.ANY, Axis.COMPONENT_OPTIONAL)],
        unsqueeze_dim: int = 1,
        dimensions_without_component: int = 1,
    ) -> Float[Tensor, Axis.names(Axis.ANY, Axis.COMPONENT)]:
        """Add component axis if missing.

        Examples:
            If the component axis is missing, add it:

            >>> import torch
            >>> input = torch.tensor([1.0, 2.0, 3.0])
            >>> AbstractMetric.add_component_axis_if_missing(input)
            tensor([[1.],
                    [2.],
                    [3.]])

            If the component axis is present, do nothing:

            >>> import torch
            >>> input = torch.tensor([[1.0], [2.0], [3.0]])
            >>> AbstractMetric.add_component_axis_if_missing(input)
            tensor([[1.],
                    [2.],
                    [3.]])

        Args:
            input_tensor: Tensor with or without a component axis.
            unsqueeze_dim: The dimension to unsqueeze the component axis.
            dimensions_without_component: The number of dimensions of the input tensor without a
                component axis.

        Returns:
            Tensor with a component axis.

        Raises:
            ValueError: If the number of dimensions of the input tensor is not supported.
        """
        if input_tensor.ndim == dimensions_without_component:
            return input_tensor.unsqueeze(unsqueeze_dim)

        if input_tensor.ndim == dimensions_without_component + 1:
            return input_tensor

        error_message = f"Unexpected number of dimensions: {input_tensor.ndim}"
        raise ValueError(error_message)


class MetricResult:
    """Metric result created by an `AbstractMetric` subclass."""

    _metric_location: MetricLocation
    _metric_name: str
    _metric_postfix: str | None
    _component_wise_values: Sequence[WandbSupportedLogTypes] | Float[
        Tensor, Axis.names(Axis.COMPONENT)
    ] | Int[Tensor, Axis.names(Axis.COMPONENT)]
    _component_names: list[str]
    _aggregate_approach: ComponentAggregationApproach | None
    _aggregate_value: Any | None

    def __init__(
        self,
        component_names: list[str] | None,
        component_wise_values: Sequence[WandbSupportedLogTypes]
        | Float[Tensor, Axis.names(Axis.COMPONENT)]
        | Int[Tensor, Axis.names(Axis.COMPONENT)],
        name: str,
        pipeline_location: MetricLocation,
        aggregate_approach: ComponentAggregationApproach
        | None = ComponentAggregationApproach.TABLE,
        aggregate_value: Any | None = None,  # noqa: ANN401
        postfix: str | None = None,
    ) -> None:
        """Initialize a metric result.

        Args:
            component_wise_values: Component-wise values.
            name: Metric name (e.g. `l2_loss`). This will be combined with
                the component name and metric locations, as well as an optional postfix, to create a
                Weights and Biases name of the form
                `component_name/metric_location/metric_name/metric_postfix`.
            pipeline_location: Metric location.
            aggregate_approach: Component aggregation approach.
            aggregate_value: Override the aggregate value across components. For most metric results
                you can instead just specify the `aggregate_approach` and it will be automatically
                calculated.
            component_names: Component names if there are multiple components.
            postfix: Metric name postfix.

        Raises:
            ValueError: If the number of component names does not match the number of component-wise
            values.
        """
        # Validate
        if component_names is not None and len(component_names) != len(component_wise_values):
            error_message = (
                f"Number of component names ({len(component_names)}) does not match the number "
                f"of component-wise values ({len(component_wise_values)})."
            )
            raise ValueError(error_message)

        self._metric_location = pipeline_location
        self._metric_name = name
        self._component_wise_values = component_wise_values
        self._aggregate_approach = aggregate_approach
        self._aggregate_value = aggregate_value
        self._metric_postfix = postfix
        self._component_names = component_names or [
            f"component_{i}" for i in range(len(component_wise_values))
        ]

    @final
    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self._component_wise_values)

    @final
    @property
    def aggregate_value(self) -> WandbSupportedLogTypes:  # noqa: PLR0911
        """Aggregate value across components.

        Returns:
            Aggregate value (defaults to the initialised aggregate value if set, or otherwise
            attempts to automatically aggregate the component-wise values).

        Raises:
            ValueError: If the component-wise values cannot be aggregated.
        """
        # Allow overriding
        if self._aggregate_value is not None:
            return self._aggregate_value

        if self.n_components == 1:
            return self._component_wise_values[0]

        cannot_aggregate_error_message = "Cannot aggregate component-wise values."

        # Automatically aggregate number lists/sequences/tuples/sets
        if (isinstance(self._component_wise_values, (Sequence, list, tuple, set))) and all(
            isinstance(x, (int, float)) for x in self._component_wise_values
        ):
            values: list = cast(list[float], self._component_wise_values)
            match self._aggregate_approach:
                case ComponentAggregationApproach.MEAN:
                    return sum(values) / len(values)
                case ComponentAggregationApproach.MAX:
                    return max(values)
                case ComponentAggregationApproach.MIN:
                    return min(values)
                case ComponentAggregationApproach.SUM:
                    return sum(values)
                case ComponentAggregationApproach.TABLE:
                    return values
                case _:
                    raise ValueError(cannot_aggregate_error_message)

        # Automatically aggregate number tensors
        if (
            isinstance(self._component_wise_values, Tensor)
            and self._component_wise_values.shape[0] == self.n_components
        ):
            match self._aggregate_approach:
                case ComponentAggregationApproach.MEAN:
                    return self._component_wise_values.mean(dim=0)
                case ComponentAggregationApproach.MAX:
                    return torch.max(self._component_wise_values, dim=-1)  # type: ignore
                case ComponentAggregationApproach.MIN:
                    return torch.min(self._component_wise_values, dim=-1)  # type: ignore
                case ComponentAggregationApproach.SUM:
                    return self._component_wise_values.sum(dim=0)
                case ComponentAggregationApproach.TABLE:
                    return self._component_wise_values
                case _:
                    raise ValueError(cannot_aggregate_error_message)

        #  Raise otherwise
        raise ValueError(cannot_aggregate_error_message)

    @final
    def create_wandb_name(self, component_name: str | None = None) -> str:
        """Weights and Biases Metric Name.

        Example:
            >>> metric_result = MetricResult(
            ...     location=MetricLocation.VALIDATE,
            ...     name="loss",
            ...     component_wise_values=[1.0, 2.0, 3.0],
            ... )
            >>> metric_result.create_wandb_name()
            'validate/loss'

            >>> metric_result.create_wandb_name(component_name="mlp_1")
            'mlp_1/validate/loss'
        """
        name_parts = []

        if component_name is not None:
            name_parts.append(component_name)

        name_parts.extend([self._metric_location.value, self._metric_name])

        if self._metric_postfix is not None:
            name_parts.append(self._metric_postfix)

        return "/".join(name_parts)

    @final
    @property
    def wandb_log(self) -> dict[str, WandbSupportedLogTypes]:
        """Create the Weights and Biases Log data.

        For use with `wandb.log()`.

        https://docs.wandb.ai/ref/python/log

        Examples:
            With just one component:

            >>> metric_result = MetricResult(
            ...     location=MetricLocation.VALIDATE,
            ...     name="loss",
            ...     component_wise_values=[1.0],
            ... )
            >>> metric_result.wandb_log
            {'validate/loss': 1.0}

            With multiple components:

            >>> metric_result = MetricResult(
            ...     location=MetricLocation.VALIDATE,
            ...     name="loss",
            ...     component_wise_values=[1.0, 2.0],
            ...     component_names=["mlp_1", "mlp_2"]
            ... )
            >>> metric_result.wandb_log
            {'mlp_1/validate/loss': 1.0, 'mlp_2/validate/loss': 2.0, 'validate/loss': 1.5}

        Args:
            component_names: Component names if there are multiple components.

        Returns:
            Weights and Biases log data.
        """
        # Create the component wise logs if there is more than one component
        component_wise_logs = {}
        if self.n_components > 1:
            for component_name, value in zip(self._component_names, self._component_wise_values):
                component_wise_logs[self.create_wandb_name(component_name=component_name)] = value

        # Create the aggregate log if there is an aggregate value
        aggregate_log = {}
        if self._aggregate_approach is not None or self._aggregate_value is not None:
            aggregate_log = {self.create_wandb_name(): self.aggregate_value}

        return {**component_wise_logs, **aggregate_log}

    def __str__(self) -> str:
        """String representation."""
        return str(self.wandb_log)

    def __repr__(self) -> str:
        """Representation."""
        class_name = self.__class__.__name__
        return f"""{class_name}(
            metric_location={self._metric_location},
            metric_name={self._metric_name},
            metric_postfix={self._metric_postfix},
            component_wise_values={self._component_wise_values},
            component_names={self._component_names},
            aggregate_approach={self._aggregate_approach},
            aggregate_value={self._aggregate_value},
        )"""


class AbstractMetric(ABC):
    """Abstract metric."""

    _component_names: list[str] | None
    """Component names."""

    @property
    @abstractmethod
    def metric_location(self) -> MetricLocation:
        """Metric location."""

    @abstractmethod
    def calculate(self, data) -> list[MetricResult]:  # type: ignore # noqa: ANN001 (type to be narrowed by abstract subclasses)
        """Calculate metrics."""

    def __init__(self, component_names: list[str] | None = None) -> None:
        """Initialise the metric.

        Args:
            component_names: Component names if there are multiple components.
        """
        self._component_names = component_names
