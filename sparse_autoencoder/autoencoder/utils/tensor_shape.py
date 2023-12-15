"""Tensor shape utilities."""


def shape_with_optional_dimensions(*shape: int | None) -> tuple[int, ...]:
    """Create a shape from a tuple of optional dimensions.

    Motivation:
        By default PyTorch tensor shapes will error if you set an axis to `None`. This allows
        you to set that size and then the resulting output simply removes that axis.

    Examples:
        >>> shape_with_optional_dimensions(1, 2, 3)
        (1, 2, 3)

        >>> shape_with_optional_dimensions(1, None, 3)
        (1, 3)

        >>> shape_with_optional_dimensions(1, None, None)
        (1,)

        >>> shape_with_optional_dimensions(None, None, None)
        ()

    Args:
        *shape: Axis sizes, with `None` representing an optional axis.

    Returns:
        Axis sizes.
    """
    return tuple(dimension for dimension in shape if dimension is not None)
