"""Util to add a component axis (dimension) if missing to a tensor."""
from torch import Tensor


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
        >>> add_component_axis_if_missing(input)
        tensor([[1.],
                [2.],
                [3.]])

        If the component axis is present, do nothing:

        >>> import torch
        >>> input = torch.tensor([[1.0], [2.0], [3.0]])
        >>> add_component_axis_if_missing(input)
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
