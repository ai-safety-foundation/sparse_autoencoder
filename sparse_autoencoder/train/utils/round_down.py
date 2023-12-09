"""Round down to the nearest multiple."""


def round_to_multiple(value: int | float, multiple: int) -> int:  # noqa: PYI041
    """Round down to the nearest multiple.

    Helper function for creating default values.

    Example:
        >>> round_to_multiple(1023, 100)
        1000

    Args:
        value: The value to round down.
        multiple: The multiple to round down to.

    Returns:
        The value rounded down to the nearest multiple.

    Raises:
        ValueError: If `value` is less than `multiple`.
    """
    int_value = int(value)

    if int_value < multiple:
        error_message = f"{value=} must be greater than or equal to {multiple=}"
        raise ValueError(error_message)

    return int_value - int_value % multiple
