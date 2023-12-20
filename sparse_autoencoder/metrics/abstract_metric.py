"""Abstract metric."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import auto
from typing import Any, TypeAlias, cast, final

from jaxtyping import Float, Int
import numpy as np
from strenum import LowercaseStrEnum, SnakeCaseStrEnum
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
        input_tensor: Tensor,
        unsqueeze_dim: int = 1,
        dimensions_without_component: int = 1,
    ) -> Tensor:
        """Add component axis if missing.

        Examples:
            If the component axis is missing, add it:

            >>> import torch
            >>> input = torch.tensor([1.0, 2.0, 3.0])
            >>> MetricInputData.add_component_axis_if_missing(input)
            tensor([[1.],
                    [2.],
                    [3.]])

            If the component axis is present, do nothing:

            >>> import torch
            >>> input = torch.tensor([[1.0], [2.0], [3.0]])
            >>> MetricInputData.add_component_axis_if_missing(input)
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
    """Metric result."""

    location: MetricLocation
    name: str
    postfix: str | None
    _component_names: list[str]
    component_wise_values: Sequence[WandbSupportedLogTypes] | Float[
        Tensor, Axis.names(Axis.COMPONENT)
    ] | Int[Tensor, Axis.names(Axis.COMPONENT)]
    aggregate_approach: ComponentAggregationApproach | None
    _aggregate_value: Any | None

    def __init__(
        self,
        component_wise_values: Sequence[WandbSupportedLogTypes]
        | Float[Tensor, Axis.names(Axis.COMPONENT)]
        | Int[Tensor, Axis.names(Axis.COMPONENT)],
        name: str,
        location: MetricLocation,
        aggregate_approach: ComponentAggregationApproach
        | None = ComponentAggregationApproach.TABLE,
        aggregate_value: Any | None = None,  # noqa: ANN401
        postfix: str | None = None,
    ) -> None:
        """Initialize a metric result.

        Args:
            component_wise_values: Values for each component.
            name: Metric name (e.g. `l2_loss`). This will be combined with the component name and
                metric locations, as well as an optional postfix, to create a Weights and Biases
                name of the form `component_name/metric_location/metric_name/metric_postfix`.
            location: Metric location.
            aggregate_approach: Component aggregation approach.
            aggregate_value: Override the aggregate value across components. For most metric results
                you can instead just specify the `aggregate_approach` and it will be automatically
                calculated.
            postfix: Metric name postfix.
        """
        self.location = location
        self.name = name
        self.component_wise_values = component_wise_values
        self.aggregate_approach = aggregate_approach
        self._aggregate_value = aggregate_value
        self.postfix = postfix
        self._component_names = [f"component_{i}" for i in range(len(component_wise_values))]

    @final
    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self.component_wise_values)

    @final
    @property
    def aggregate_value(  # noqa: PLR0911
        self,
    ) -> (
        WandbSupportedLogTypes
        | Float[Tensor, Axis.names(Axis.COMPONENT)]
        | Int[Tensor, Axis.names(Axis.COMPONENT)]
    ):
        """Aggregate value across components.

        Returns:
            Aggregate value (defaults to the initialised aggregate value if set, or otherwise
            attempts to automatically aggregate the component-wise values).

        Raises:
            ValueError: If the component-wise values cannot be automatically aggregated.
        """
        # Allow overriding
        if self._aggregate_value is not None:
            return self._aggregate_value

        if self.n_components == 1:
            return self.component_wise_values[0]

        cannot_aggregate_error_message = "Cannot aggregate component-wise values."

        # Automatically aggregate number lists/sequences/tuples/sets
        if (isinstance(self.component_wise_values, (Sequence, list, tuple, set))) and all(
            isinstance(x, (int, float)) for x in self.component_wise_values
        ):
            values: list = cast(list[float], self.component_wise_values)
            match self.aggregate_approach:
                case ComponentAggregationApproach.MEAN:
                    return sum(values) / len(values)
                case ComponentAggregationApproach.SUM:
                    return sum(values)
                case ComponentAggregationApproach.TABLE:
                    return values
                case _:
                    raise ValueError(cannot_aggregate_error_message)

        # Automatically aggregate number tensors
        if (
            isinstance(self.component_wise_values, Tensor)
            and self.component_wise_values.shape[0] == self.n_components
        ):
            match self.aggregate_approach:
                case ComponentAggregationApproach.MEAN:
                    return self.component_wise_values.mean(dim=0)
                case ComponentAggregationApproach.SUM:
                    return self.component_wise_values.sum(dim=0)
                case ComponentAggregationApproach.TABLE:
                    return self.component_wise_values
                case _:
                    raise ValueError(cannot_aggregate_error_message)

        #  Raise otherwise
        raise ValueError(cannot_aggregate_error_message)

    @final
    def create_wandb_name(self, component_name: str | None = None) -> str:
        """Weights and Biases Metric Name.

        Note Weights and Biases categorises metrics using a forward slash (`/`) in the name string.

        Example:
            >>> metric_result = MetricResult(
            ...     location=MetricLocation.VALIDATE,
            ...     name="loss",
            ...     component_wise_values=[1.0, 2.0, 3.0],
            ... )
            >>> metric_result.create_wandb_name()
            'validate/loss'

            >>> metric_result.create_wandb_name(component_name="component_0")
            'component_0/validate/loss'

        Args:
            component_name: Component name, if creating a Weights and Biases name for a specific
                component.

        Returns:
            Weights and Biases metric name.
        """
        name_parts = []

        if component_name is not None:
            name_parts.append(component_name)

        name_parts.extend([self.location.value, self.name])

        if self.postfix is not None:
            name_parts.append(self.postfix)

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
            ...     component_wise_values=[1.5],
            ... )
            >>> for k, v in metric_result.wandb_log.items():
            ...     print(f"{k}: {v}")
            validate/loss: 1.5

            With multiple components:

            >>> metric_result = MetricResult(
            ...     location=MetricLocation.VALIDATE,
            ...     name="loss",
            ...     component_wise_values=[1.0, 2.0],
            ...     aggregate_approach=ComponentAggregationApproach.MEAN,
            ... )
            >>> for k, v in metric_result.wandb_log.items():
            ...     print(f"{k}: {v}")
            component_0/validate/loss: 1.0
            component_1/validate/loss: 2.0
            validate/loss: 1.5

        Returns:
            Weights and Biases log data.
        """
        # Create the component wise logs if there is more than one component
        component_wise_logs = {}
        if self.n_components > 1:
            for component_name, value in zip(self._component_names, self.component_wise_values):
                component_wise_logs[self.create_wandb_name(component_name=component_name)] = value

        # Create the aggregate log if there is an aggregate value
        aggregate_log = {}
        if self.aggregate_approach is not None or self._aggregate_value is not None:
            aggregate_log = {self.create_wandb_name(): self.aggregate_value}

        return {**component_wise_logs, **aggregate_log}

    def __str__(self) -> str:
        """String representation."""
        return str(self.wandb_log)

    def __repr__(self) -> str:
        """Representation."""
        class_name = self.__class__.__name__
        return f"""{class_name}(
            location={self.location},
            name={self.name},
            postfix={self.postfix},
            component_wise_values={self.component_wise_values},
            aggregate_approach={self.aggregate_approach},
            aggregate_value={self._aggregate_value},
        )"""


class AbstractMetric(ABC):
    """Abstract metric."""

    @property
    @abstractmethod
    def location(self) -> MetricLocation:
        """Metric location."""

    @abstractmethod
    def calculate(self, data) -> list[MetricResult]:  # type: ignore # noqa: ANN001 (type to be narrowed by abstract subclasses)
        """Calculate metrics."""
