"""Methods to reshape activation tensors."""
from collections.abc import Callable
from functools import reduce
from typing import Any, TypeAlias, cast

from jaxtyping import Float
from torch import Tensor

from sparse_autoencoder.tensor_types import Axis


ReshapeActivationsFunction: TypeAlias = Callable[
    [Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)]],
    Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)],
]
"""Reshape Activations Function.

Used within hooks to e.g. reshape activations before storing them in the activation store.
"""


def reshape_to_last_dimension(
    batch_activations: Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)],
) -> Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)]:
    """Reshape to Last Dimension.

    Takes a tensor of activation vectors, with arbitrary numbers of dimensions (the last of which is
    the neurons dimension), and returns a single tensor of size [item, neurons].

    Examples:
    With 2 axis (e.g. pos neuron):

    >>> import torch
    >>> input = torch.rand(3, 100)
    >>> res = reshape_to_last_dimension(input)
    >>> res.shape
    torch.Size([3, 100])

    With 3 axis (e.g. batch, pos, neuron):

    >>> input = torch.randn(3, 3, 100)
    >>> res = reshape_to_last_dimension(input)
    >>> res.shape
    torch.Size([9, 100])

    With 4 axis (e.g. batch, pos, head_idx, neuron)

    >>> input = torch.rand(3, 3, 3, 100)
    >>> res = reshape_to_last_dimension(input)
    >>> res.shape
    torch.Size([27, 100])

    Args:
        batch_activations: Input Activation Store Batch

    Returns:
        Single Tensor of Activation Store Items
    """
    # typecast batch_activations to Any (i.e. ignore type information)
    batch_activations = cast(Any, batch_activations)

    return batch_activations.reshape(-1, batch_activations.shape[-1])


def reshape_concat_last_dimensions(
    batch_activations: Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)],
    concat_dims: int,
) -> Float[Tensor, Axis.names(Axis.ANY, Axis.INPUT_OUTPUT_FEATURE)]:
    """Reshape to Last Dimension, Concatenating the Specified Dimensions.

    Takes a tensor of activation vectors, with arbitrary numbers of dimensions (the last
    `concat_dims` of which are the neuron dimensions), and returns a single tensor of size
    [item, neurons].

    Args:
        batch_activations: Input Activation Store Batch
        concat_dims: Number of dimensions to concatenate

    Returns:
        Single Tensor of Activation Store Items
    """
    # typecast batch_activations to Any (i.e. ignore type information)
    batch_activations = cast(Any, batch_activations)

    neurons = reduce(lambda x, y: x * y, batch_activations.shape[-concat_dims:])
    items = reduce(lambda x, y: x * y, batch_activations.shape[:-concat_dims])

    return batch_activations.reshape(items, neurons)
