"""Component slice tensor utils."""
from torch import Tensor


def get_component_slice_tensor(
    input_tensor: Tensor,
    n_dim_with_component: int,
    component_dim: int,
    component_idx: int,
) -> Tensor:
    """Get a slice of a tensor for a specific component.

    Examples:
        >>> import torch
        >>> input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> get_component_slice_tensor(input_tensor, 2, 1, 0)
        tensor([1, 3, 5, 7])

        >>> input_tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        >>> get_component_slice_tensor(input_tensor, 3, 1, 0)
        tensor([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])

    Args:
        input_tensor: Input tensor.
        n_dim_with_component: Number of dimensions in the input tensor with the component axis.
        component_dim: Dimension of the component axis.
        component_idx: Index of the component to get the slice for.

    Returns:
        Tensor slice.

    Raises:
        ValueError: If the input tensor does not have the expected number of dimensions.
    """
    if n_dim_with_component - 1 == input_tensor.ndim:
        return input_tensor

    if n_dim_with_component != input_tensor.ndim:
        error_message = (
            f"Cannot get component slice for tensor with {input_tensor.ndim} dimensions "
            f"and {n_dim_with_component} dimensions with component."
        )
        raise ValueError(error_message)

    # Create a tuple of slices for each dimension
    slice_tuple = tuple(
        component_idx if i == component_dim else slice(None) for i in range(input_tensor.ndim)
    )

    return input_tensor[slice_tuple]
