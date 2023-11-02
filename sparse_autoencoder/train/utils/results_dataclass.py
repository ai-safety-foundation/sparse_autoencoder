"""Results dataclass utilities."""
from dataclasses import MISSING, field, fields, make_dataclass
from typing import Any, Type, get_args, get_origin

from sparse_autoencoder.train.utils.wandb_sweep_types import Parameter


def convert_parameters_to_results_type(config_dataclass: Type) -> Type:
    """Convert Parameters to a Results Dataclass Type

    Converts a :class:`sparse_autoencoder.train.utils.wandb_sweep_types.Parameters` dataclass type
    signature into a parameter results dataclass type signature.

    Example:

    >>> from dataclasses import dataclass, field
    >>> @dataclass
    ... class SweepParameterConfig:
    ...
    ...     lr: Parameter[float] = field(default_factory=lambda: Parameter(value=0.001))
    ...
    ...     lr_list: Parameter[float] = field(default_factory=lambda: Parameter(value=[0.002, 0.004]))
    ...
    >>> SweepParameterResults = convert_parameters_to_results_type(SweepParameterConfig)
    >>> SweepParameterResults.__annotations__['lr']
    <class 'float'>

    >>> SweepParameterResults.__annotations__['lr_list']
    <class 'float'>

    Args:
        config_dataclass: The config dataclass to convert.
    """
    new_fields: list[tuple[str, Any, Any]] = []

    for f in fields(config_dataclass):
        # Determine if the default should come from a default or a factory
        if f.default is not MISSING:
            default_value = f.default
        elif f.default_factory is not MISSING:  # Use the default factory if provided
            default_value = field(  # pylint: disable=invalid-field-call
                default_factory=f.default_factory
            )
        else:
            default_value = MISSING

        # If the field is a Parameter, replace it with the contained type
        if get_origin(f.type) == Parameter:
            contained_type = get_args(f.type)[0]
            # If the contained type is a list, go one level deeper
            if get_origin(contained_type) == list:
                list_contained_type = get_args(contained_type)[0]
                new_fields.append((f.name, list[list_contained_type], default_value))
            else:
                new_fields.append((f.name, contained_type, default_value))
        else:
            new_fields.append((f.name, f.type, default_value))

    # Create a new dataclass with the new fields
    return make_dataclass(config_dataclass.__name__ + "Results", new_fields)
